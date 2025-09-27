# --- Installation ---
# pip install langchain sentence-transformers httpx qdrant-client langchain-community sqlalchemy pandas langchain-core requests langchain_google_genai duckduckgo-search langgraph pydantic

import json
import asyncio
import qdrant_client
import sqlite3
import pandas as pd
import os
from langchain.tools import tool
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, TypedDict, Optional

# --- Pydantic for Structured Output ---
from pydantic.v1 import BaseModel, Field

# --- Official LangChain & Google Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.tools import DuckDuckGoSearchRun

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# --- API KEY CONFIGURATION ---
API_KEY = "AIzaSyC64jh0jHd6OegJcXsfMpp__nm4kLBH1dg"  # <-- PASTE YOUR KEY HERE

if API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    print("FATAL: Please replace 'YOUR_GOOGLE_API_KEY_HERE' with your actual Google API key.")
    exit()

# --- MODEL NAME CONFIGURATION ---
# Use the stable, universally available model name.
MODEL_NAME = "gemini-2.5-flash-lite"

# --- Configuration ---
DATA_DIR = "data"
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_db")
DB_PATH = os.path.join(DATA_DIR, "financials.db")
TABLE_NAME = "financial_summary"
STRUCTURED_CSV_PATH = os.path.join(DATA_DIR, "alphabet_financials_structured.csv")
ENRICHED_CHUNKS_PATH = os.path.join(DATA_DIR, "enriched_chunks.json")
COLLECTION_NAME = "financial_docs_alphabet"

# --- Robust Setup Function ---
def setup_data_stores():
    print("--- Verifying Data Stores ---"); os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}';")
    if not cursor.fetchone():
        print(f"Table '{TABLE_NAME}' not found. Creating it now...")
        if not os.path.exists(STRUCTURED_CSV_PATH): print(f"FATAL: Structured data file missing."); conn.close(); exit()
        df = pd.read_csv(STRUCTURED_CSV_PATH)
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        print(f"Successfully created table '{TABLE_NAME}'.")
    else: print(f"Table '{TABLE_NAME}' found in SQLite database.")
    conn.close()
    if not os.path.exists(ENRICHED_CHUNKS_PATH): print(f"FATAL: Enriched chunks file not found."); exit()
    print("Enriched chunks file found."); print("--- Data Store verification complete. ---\n")

# --- Initialize Models & Components ---
setup_data_stores()
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY, convert_system_message_to_human=True)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
client = qdrant_client.QdrantClient(path=QDRANT_PATH)
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
sql_agent_executor = create_sql_agent(llm=llm, toolkit=SQLDatabaseToolkit(db=db, llm=llm), verbose=False, agent_type="openai-tools", handle_parsing_errors=True)

# --- Specialist Tools ---
async def optimize_query(query: str) -> str:
    prompt = f'Rewrite the following query for a semantic search system. Respond with ONLY the rewritten query. Query: "{query}" Rewritten:'
    return (await llm.ainvoke(prompt)).content
@tool
async def librarian_rag_tool(query: str) -> List[Dict[str, Any]]:
    """Retrieves deep, contextual information from Alphabet's financial filings."""
    print(f"\n-- Librarian: Searching for '{query}' --")
    optimized_query = await optimize_query(query)
    query_embedding = embedding_model.embed_query(optimized_query)
    search_results = client.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=20, with_payload=True)
    rerank_pairs = [[optimized_query, r.payload['content']] for r in search_results]
    scores = await asyncio.to_thread(cross_encoder_model.predict, rerank_pairs)
    for i, s in enumerate(scores): search_results[i].score = s
    reranked_results = sorted(search_results, key=lambda x: x.score, reverse=True)
    return [{'source': r.payload['source'], 'content': r.payload['content'], 'summary': r.payload['summary']} for r in reranked_results[:5]]
@tool
async def analyst_sql_tool(query: str) -> str:
    """Answers specific questions about revenue, net income, etc., from a SQL database."""
    print(f"\n-- SQL Analyst: Querying SQL for '{query}' --")
    try: return (await sql_agent_executor.ainvoke({"input": query})).get("output", "No answer.")
    except Exception as e: print(f"SQL agent error: {e}"); return "Error."
def _analyst_trend_tool_sync(query: str) -> str:
    print(f"\n-- Trend Analyst: Analyzing '{query}' --")
    conn = sqlite3.connect(DB_PATH); df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY year, quarter", conn); conn.close();
    df['period'] = df['year'].astype(str) + '-' + df['quarter']; df.set_index('period', inplace=True); metric = 'revenue_usd_billions'
    df['QoQ_Growth'] = df[metric].pct_change(); df['YoY_Growth'] = df[metric].pct_change(4)
    start_val, latest_val = df[metric].iloc[0], df[metric].iloc[-1]
    latest_qoq, latest_yoy = df['YoY_Growth'].iloc[-1], df['QoQ_Growth'].iloc[-1]
    return f"Analysis of {metric} from {df.index[0]} to {df.index[-1]}: Trend from ${start_val:.1f}B to ${latest_val:.1f}B. Recent QoQ: {latest_yoy:.1%}, recent YoY: {latest_qoq:.1%}. Consistent growth shown."
@tool
async def analyst_trend_tool(query: str) -> str:
    """Analyzes financial trends like QoQ or YoY growth from the SQL database."""
    return await asyncio.to_thread(_analyst_trend_tool_sync, query)
@tool
async def scout_web_search_tool(query: str) -> str:
    """Searches the web for real-time information like stock prices or recent news."""
    print(f"\n-- Scout: Searching web for '{query}' --")
    search = DuckDuckGoSearchRun()
    return await search.arun(query)

# --- PHASE 3: AGENTIC SYSTEM ---
class AgentState(TypedDict):
    original_request: str; clarification_question: Optional[str]; plan: List[Dict[str, Any]]; intermediate_steps: List[Dict[str, Any]]; verification_history: List[Dict[str, Any]]; final_response: str
tools = [librarian_rag_tool, analyst_sql_tool, analyst_trend_tool, scout_web_search_tool]
tool_map = {tool.name: tool for tool in tools}

class GatekeeperDecision(BaseModel):
    is_specific: bool = Field(description="True if the request is specific, False if ambiguous."); clarification_question: Optional[str] = Field(description="The question to ask if ambiguous.")
gatekeeper_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY).with_structured_output(GatekeeperDecision)
async def ambiguity_check_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Gatekeeper (Ambiguity Check) Node --"); request = state['original_request']
    prompt = f"You are a strict Gatekeeper. A specific request has actionable keywords (revenue, risk, trend, year). An ambiguous request is vague ('How is the company?'). A request asking for multiple things or a comparison IS SPECIFIC. Based on these rules, is the following request specific? Request: \"{request}\""
    decision = await gatekeeper_llm.ainvoke(prompt)
    if decision.is_specific: print("  - Request is specific."); return {"clarification_question": None}
    else: print("  - Request is ambiguous."); return {"clarification_question": decision.clarification_question}

# *** THE CRITICAL FIX IS HERE: A SMARTER PLANNER WITH MEMORY ***
class PlanStep(BaseModel):
    tool_name: str = Field(description="The name of the tool to call."); tool_input: str = Field(description="The full, complete string input for the tool.")
class Plan(BaseModel):
    steps: List[PlanStep] = Field(description="A list of tool calls to execute. The final step must be a call to a tool named 'FINISH'.")
planner_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY).with_structured_output(Plan)
def create_planner_prompt(tools):
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description.strip()}" for tool in tools]) + "\n- FINISH: Use this tool to signal that you have a complete answer."
    # The prompt now includes a section for feedback on previous failed attempts.
    return f"""Create a step-by-step plan to answer the user's request using the available tools. The final step must ALWAYS be to call the 'FINISH' tool.

**Tools:**
{tool_descriptions}

{{feedback}}

User Request: {{request}}
"""
planner_prompt_template = create_planner_prompt(tools)
async def planner_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Planner Node --")
    request = state['original_request']
    feedback_str = ""
    # If there is feedback from the Auditor, format it for the Planner.
    if state.get("verification_history"):
        feedback = state["verification_history"][-1]
        failed_step = state["intermediate_steps"][-1]
        feedback_str = f"""**Previous Attempt Feedback:**
The last plan failed. The tool '{failed_step['tool_name']}' was called, but the auditor gave a low confidence score of {feedback['confidence_score']}/5. 
The auditor's reasoning was: '{feedback['reasoning']}'.
Based on this feedback, you MUST create a new, different plan to better address the user's full request. Do not repeat the same first step if it was deemed insufficient."""
    
    prompt = planner_prompt_template.format(request=request, feedback=feedback_str)
    plan = await planner_llm.ainvoke(prompt)
    print(f"  - Generated Plan: {[step.tool_name for step in plan.steps]}")
    return {"plan": [step.dict() for step in plan.steps]}

async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Tool Executor Node --"); plan_step = state['plan'][0]; tool_name = plan_step['tool_name']; tool_input = plan_step['tool_input']
    if tool_name == "FINISH": return {"plan": state['plan'][1:]}
    tool_to_call = tool_map[tool_name]
    print(f"  - Executing: {tool_name} with input '{tool_input}'")
    result = await tool_to_call.ainvoke({"query": tool_input})
    new_step = {'tool_name': tool_name, 'tool_input': tool_input, 'tool_output': result}
    return {"intermediate_steps": state.get('intermediate_steps', []) + [new_step], "plan": state['plan'][1:]}

class VerificationResult(BaseModel):
    confidence_score: int = Field(description="Score 1-5 on confidence."); is_relevant: bool; reasoning: str
auditor_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=API_KEY).with_structured_output(VerificationResult)
async def verification_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Auditor (Self-Correction) Node --")
    if not state.get('intermediate_steps'): print("  - No steps to audit. Skipping."); return {}
    request = state['original_request']; last_step = state['intermediate_steps'][-1]
    prompt = f"Audit the output from a tool.\n**User Request:** {request}\n**Tool:** {last_step['tool_name']}\n**Tool Output:** {json.dumps(last_step['tool_output'])}\n\n**Audit Checklist:**\n1. Relevance (Score 1-5)\n2. Consistency"
    audit_result = await auditor_llm.ainvoke(prompt)
    print(f"  - Audit Confidence Score: {audit_result.confidence_score}/5. Reasoning: {audit_result.reasoning}")
    return {"verification_history": state.get('verification_history', []) + [audit_result.dict()]}

synthesizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=API_KEY, temperature=0.2)
async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Strategist (Synthesizer) Node --"); request = state['original_request']
    context = "\n\n".join([f"## Tool: {s['tool_name']}\nOutput: {json.dumps(s['tool_output'])}" for s in state['intermediate_steps']])
    prompt = f"You are an expert financial analyst. Synthesize a comprehensive answer for the user's request based on the context. Infer potential causal links and frame them as a data-grounded hypothesis.\n\n**User Request:**\n{request}\n\n**Context:**\n---\n{context}\n---\n\nFinal Answer:"
    final_answer = await synthesizer_llm.ainvoke(prompt)
    print("  - Generated final answer with causal inference.")
    return {"final_response": final_answer.content}

def router_node(state: AgentState) -> str:
    print("\n-- Advanced Router Node --")
    if state.get("clarification_question"): print("  - Decision: Ambiguity detected. Halting."); return END
    # If the plan is empty, this is the first pass after the Gatekeeper.
    if not state.get("plan"):
        print("  - Decision: New request. Routing to planner.")
        return "planner"
    # If we have feedback, it means a tool failed. Go back to the planner.
    if state.get("verification_history"):
        if state["verification_history"][-1]["confidence_score"] < 3:
            print(f"  - Decision: Verification failed. Re-planning.")
            # **CRITICAL FIX**: Clear the intermediate steps so the synthesizer doesn't see failed results.
            state['intermediate_steps'] = []
            return "planner"
    # If the plan is finished, go to the synthesizer.
    if not state.get("plan") or state["plan"][0]['tool_name'] == "FINISH":
        print("  - Decision: Plan complete. Routing to synthesizer.")
        return "synthesize"
    # Otherwise, continue the plan.
    else:
        print("  - Decision: Plan has more steps. Routing to tool executor.")
        return "execute_tool"

# --- Build the Graph ---
graph_builder = StateGraph(AgentState)
graph_builder.add_node("ambiguity_check", ambiguity_check_node); graph_builder.add_node("planner", planner_node); graph_builder.add_node("execute_tool", tool_executor_node); graph_builder.add_node("verify", verification_node); graph_builder.add_node("synthesize", synthesizer_node)
graph_builder.set_entry_point("ambiguity_check")
graph_builder.add_conditional_edges("ambiguity_check", lambda state: "planner" if state.get("clarification_question") is None else END)
graph_builder.add_edge("planner", "execute_tool"); graph_builder.add_edge("execute_tool", "verify")
graph_builder.add_conditional_edges("verify", router_node, {"planner": "planner", "execute_tool": "execute_tool", "synthesize": "synthesize"})
graph_builder.add_edge("synthesize", END)
agentic_system = graph_builder.compile()
print("Advanced Agentic System compiled successfully!")

# --- Main Execution Logic ---
async def run_agentic_system(query: str):
    print(f"\n\n{'='*80}\n--- Running Agentic System with Query ---\nQuery: {query}\n{'='*80}")
    inputs = {"original_request": query}
    # Increased recursion limit as a safeguard for complex queries
    final_state = await agentic_system.ainvoke(inputs, {"recursion_limit": 15})
    if final_state.get('clarification_question'):
        print("\n--- FINAL RESPONSE (CLARIFICATION) ---"); print(final_state['clarification_question'])
    else:
        print("\n--- FINAL SYNTHESIZED RESPONSE ---"); print(final_state['final_response'])

async def main():
    await run_agentic_system("Tell me about Alphabet's performance.")
    await run_agentic_system("Analyze Alphabet's revenue trend for the last two years and discuss how it might relate to the competitive risks mentioned in their latest 10-K.")

if __name__ == "__main__":
    try: asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError): print("\nExecution interrupted.")









































# # --- Installation ---
# # pip install langchain sentence-transformers httpx qdrant-client langchain-community sqlalchemy pandas langchain-core requests langchain_google_genai duckduckgo-search langgraph pydantic

# import json
# import asyncio
# import qdrant_client
# import sqlite3
# import pandas as pd
# import os
# from langchain.tools import tool
# from sentence_transformers import CrossEncoder
# from typing import List, Dict, Any, TypedDict, Optional

# # --- Pydantic for Structured Output ---
# from pydantic.v1 import BaseModel, Field

# # --- Official LangChain & Google Imports ---
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_community.agent_toolkits.sql.base import create_sql_agent
# from langchain_community.tools import DuckDuckGoSearchRun

# # --- LangGraph Imports ---
# from langgraph.graph import StateGraph, END

# # --- API KEY CONFIGURATION ---
# API_KEY = "AIzaSyC64jh0jHd6OegJcXsfMpp__nm4kLBH1dg "#"AIzaSyBbsYPL_IG4lPkAlXL6AG0Q0SEi1fbVnHs"  # <-- PASTE YOUR KEY HERE

# if API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
#     print("FATAL: Please replace 'YOUR_GOOGLE_API_KEY_HERE' with your actual Google API key.")
#     exit()

# # --- Configuration ---
# DATA_DIR = "data"
# QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_db")
# DB_PATH = os.path.join(DATA_DIR, "financials.db")
# TABLE_NAME = "financial_summary"
# STRUCTURED_CSV_PATH = os.path.join(DATA_DIR, "alphabet_financials_structured.csv")
# ENRICHED_CHUNKS_PATH = os.path.join(DATA_DIR, "enriched_chunks.json")
# COLLECTION_NAME = "financial_docs_alphabet"

# # --- Robust Setup Function ---
# def setup_data_stores():
#     print("--- Verifying Data Stores ---")
#     os.makedirs(DATA_DIR, exist_ok=True)
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}';")
#     if not cursor.fetchone():
#         print(f"Table '{TABLE_NAME}' not found. Creating it now...")
#         if not os.path.exists(STRUCTURED_CSV_PATH):
#             print(f"FATAL: Structured data file missing."); conn.close(); exit()
#         df = pd.read_csv(STRUCTURED_CSV_PATH)
#         df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
#         print(f"Successfully created table '{TABLE_NAME}'.")
#     else:
#         print(f"Table '{TABLE_NAME}' found in SQLite database.")
#     conn.close()
#     if not os.path.exists(ENRICHED_CHUNKS_PATH):
#         print(f"FATAL: Enriched chunks file not found."); exit()
#     print("Enriched chunks file found.")
#     print("--- Data Store verification complete. ---\n")

# # --- Initialize Models & Components ---
# setup_data_stores()
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY, convert_system_message_to_human=True)
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
# client = qdrant_client.QdrantClient(path=QDRANT_PATH)
# cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
# sql_agent_executor = create_sql_agent(llm=llm, toolkit=SQLDatabaseToolkit(db=db, llm=llm), verbose=False, agent_type="openai-tools", handle_parsing_errors=True)

# # --- Specialist Tools ---
# async def optimize_query(query: str) -> str:
#     prompt = f'Rewrite the following query for a semantic search system. Respond with ONLY the rewritten query. Query: "{query}" Rewritten:'
#     return (await llm.ainvoke(prompt)).content

# @tool
# async def librarian_rag_tool(query: str) -> List[Dict[str, Any]]:
#     """Retrieves deep, contextual information from Alphabet's financial filings."""
#     print(f"\n-- Librarian: Searching for '{query}' --")
#     optimized_query = await optimize_query(query)
#     query_embedding = embedding_model.embed_query(optimized_query)
#     search_results = client.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=20, with_payload=True)
#     # The cross-encoder is a CPU-bound task, so we run it in a thread to avoid blocking the event loop
#     rerank_pairs = [[optimized_query, r.payload['content']] for r in search_results]
#     scores = await asyncio.to_thread(cross_encoder_model.predict, rerank_pairs)
#     for i, s in enumerate(scores): search_results[i].score = s
#     reranked_results = sorted(search_results, key=lambda x: x.score, reverse=True)
#     return [{'source': r.payload['source'], 'content': r.payload['content'], 'summary': r.payload['summary']} for r in reranked_results[:5]]

# @tool
# async def analyst_sql_tool(query: str) -> str:
#     """Answers specific questions about revenue, net income, etc., from a SQL database."""
#     print(f"\n-- SQL Analyst: Querying SQL for '{query}' --")
#     try: return (await sql_agent_executor.ainvoke({"input": query})).get("output", "No answer.")
#     except Exception as e: print(f"SQL agent error: {e}"); return "Error."

# def _analyst_trend_tool_sync(query: str) -> str:
#     """Synchronous core logic for the trend analyst tool."""
#     print(f"\n-- Trend Analyst: Analyzing '{query}' --")
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY year, quarter", conn)
#     conn.close()
#     df['period'] = df['year'].astype(str) + '-' + df['quarter']
#     df.set_index('period', inplace=True)
#     metric = 'revenue_usd_billions'
#     df['QoQ_Growth'] = df[metric].pct_change()
#     df['YoY_Growth'] = df[metric].pct_change(4)
#     start_val, latest_val = df[metric].iloc[0], df[metric].iloc[-1]
#     latest_qoq, latest_yoy = df['YoY_Growth'].iloc[-1], df['QoQ_Growth'].iloc[-1]
#     return f"Analysis of {metric} from {df.index[0]} to {df.index[-1]}: Trend from ${start_val:.1f}B to ${latest_val:.1f}B. Recent QoQ: {latest_yoy:.1%}, recent YoY: {latest_qoq:.1%}. Consistent growth shown."

# @tool
# async def analyst_trend_tool(query: str) -> str:
#     """Analyzes financial trends like QoQ or YoY growth from the SQL database."""
#     return await asyncio.to_thread(_analyst_trend_tool_sync, query)

# @tool
# async def scout_web_search_tool(query: str) -> str:
#     """Searches the web for real-time information like stock prices or recent news."""
#     print(f"\n-- Scout: Searching web for '{query}' --")
#     search = DuckDuckGoSearchRun()
#     return await search.arun(query)

# # --- PHASE 3: AGENTIC SYSTEM ---
# class AgentState(TypedDict):
#     original_request: str; clarification_question: Optional[str]; plan: List[Dict[str, Any]]; intermediate_steps: List[Dict[str, Any]]; verification_history: List[Dict[str, Any]]; final_response: str

# tools = [librarian_rag_tool, analyst_sql_tool, analyst_trend_tool, scout_web_search_tool]
# tool_map = {tool.name: tool for tool in tools}

# class GatekeeperDecision(BaseModel):
#     is_specific: bool = Field(description="True if the request is specific, False if ambiguous."); clarification_question: Optional[str] = Field(description="The question to ask if ambiguous.")
# gatekeeper_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY).with_structured_output(GatekeeperDecision)
# async def ambiguity_check_node(state: AgentState) -> Dict[str, Any]:
#     print("\n-- Gatekeeper (Ambiguity Check) Node --"); request = state['original_request']
#     prompt = f"You are a strict Gatekeeper. A specific request has actionable keywords (revenue, risk, trend, year). An ambiguous request is vague ('How is the company?'). A request asking for multiple things or a comparison IS SPECIFIC. Based on these rules, is the following request specific? Request: \"{request}\""
#     decision = await gatekeeper_llm.ainvoke(prompt)
#     if decision.is_specific: print("  - Request is specific."); return {"clarification_question": None}
#     else: print("  - Request is ambiguous."); return {"clarification_question": decision.clarification_question}

# class PlanStep(BaseModel):
#     tool_name: str = Field(description="The name of the tool to call."); tool_input: str = Field(description="The full, complete string input for the tool.")
# class Plan(BaseModel):
#     steps: List[PlanStep] = Field(description="A list of tool calls to execute. The final step must be a call to a tool named 'FINISH'.")
# planner_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY).with_structured_output(Plan)
# def create_planner_prompt(tools):
#     tool_descriptions = "\n".join([f"- {tool.name}: {tool.description.strip()}" for tool in tools]) + "\n- FINISH: Use this tool to signal that you have a complete answer."
#     return f"Create a step-by-step plan to answer the user's request using the available tools. For each step, provide the tool name and the exact input string for that tool. The final step must ALWAYS be to call the 'FINISH' tool with an empty input.\n\n**Tools:**\n{tool_descriptions}\n\nUser Request: {{request}}"
# planner_prompt_template = create_planner_prompt(tools)
# async def planner_node(state: AgentState) -> Dict[str, Any]:
#     print("\n-- Planner Node --"); request = state['original_request']; prompt = planner_prompt_template.format(request=request)
#     plan = await planner_llm.ainvoke(prompt)
#     print(f"  - Generated Plan: {[step.tool_name for step in plan.steps]}")
#     return {"plan": [step.dict() for step in plan.steps]}

# async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
#     print("\n-- Tool Executor Node --"); plan_step = state['plan'][0]; tool_name = plan_step['tool_name']; tool_input = plan_step['tool_input']
#     if tool_name == "FINISH": return {"plan": state['plan'][1:]}
#     tool_to_call = tool_map[tool_name]
#     print(f"  - Executing: {tool_name} with input '{tool_input}'")
#     result = await tool_to_call.ainvoke({"query": tool_input})
#     new_step = {'tool_name': tool_name, 'tool_input': tool_input, 'tool_output': result}
#     return {"intermediate_steps": state.get('intermediate_steps', []) + [new_step], "plan": state['plan'][1:]}

# class VerificationResult(BaseModel):
#     confidence_score: int = Field(description="Score 1-5 on confidence."); is_relevant: bool; reasoning: str
# auditor_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY).with_structured_output(VerificationResult)
# async def verification_node(state: AgentState) -> Dict[str, Any]:
#     print("\n-- Auditor (Self-Correction) Node --")
#     if not state.get('intermediate_steps'): print("  - No steps to audit. Skipping."); return {}
#     request = state['original_request']; last_step = state['intermediate_steps'][-1]
#     prompt = f"Audit the output from a tool.\n**User Request:** {request}\n**Tool:** {last_step['tool_name']}\n**Tool Output:** {json.dumps(last_step['tool_output'])}\n\n**Audit Checklist:**\n1. Relevance (Score 1-5)\n2. Consistency"
#     audit_result = await auditor_llm.ainvoke(prompt)
#     print(f"  - Audit Confidence Score: {audit_result.confidence_score}/5")
#     return {"verification_history": state.get('verification_history', []) + [audit_result.dict()]}

# synthesizer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=API_KEY, temperature=0.2)
# async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
#     print("\n-- Strategist (Synthesizer) Node --"); request = state['original_request']
#     context = "\n\n".join([f"## Tool: {s['tool_name']}\nOutput: {json.dumps(s['tool_output'])}" for s in state['intermediate_steps']])
#     prompt = f"You are an expert financial analyst. Synthesize a comprehensive answer for the user's request based on the context. Infer potential causal links and frame them as a data-grounded hypothesis.\n\n**User Request:**\n{request}\n\n**Context:**\n---\n{context}\n---\n\nFinal Answer:"
#     final_answer = await synthesizer_llm.ainvoke(prompt)
#     print("  - Generated final answer with causal inference.")
#     return {"final_response": final_answer.content}

# def router_node(state: AgentState) -> str:
#     print("\n-- Advanced Router Node --")
#     if state.get("clarification_question"): print("  - Decision: Ambiguity detected. Halting."); return END
#     if not state.get("plan"): print("  - Decision: New request. Routing to planner."); return "planner"
#     if state.get("verification_history"):
#         if state["verification_history"][-1]["confidence_score"] < 3: print("  - Decision: Verification failed. Re-planning."); return "planner"
#     if not state.get("plan") or state["plan"][0]['tool_name'] == "FINISH": print("  - Decision: Plan complete. Routing to synthesizer."); return "synthesize"
#     else: print("  - Decision: Plan has more steps. Routing to tool executor."); return "execute_tool"

# # --- Build the Graph ---
# graph_builder = StateGraph(AgentState)
# graph_builder.add_node("ambiguity_check", ambiguity_check_node); graph_builder.add_node("planner", planner_node); graph_builder.add_node("execute_tool", tool_executor_node); graph_builder.add_node("verify", verification_node); graph_builder.add_node("synthesize", synthesizer_node)
# graph_builder.set_entry_point("ambiguity_check")
# graph_builder.add_conditional_edges("ambiguity_check", lambda state: "planner" if state.get("clarification_question") is None else END)
# graph_builder.add_edge("planner", "execute_tool"); graph_builder.add_edge("execute_tool", "verify")
# graph_builder.add_conditional_edges("verify", router_node, {"planner": "planner", "execute_tool": "execute_tool", "synthesize": "synthesize"})
# graph_builder.add_edge("synthesize", END)
# agentic_system = graph_builder.compile()
# print("Advanced Agentic System compiled successfully!")

# # --- Main Execution Logic ---
# async def run_agentic_system(query: str):
#     print(f"\n\n{'='*80}\n--- Running Agentic System with Query ---\nQuery: {query}\n{'='*80}")
#     inputs = {"original_request": query}
#     final_state = await agentic_system.ainvoke(inputs)
#     if final_state.get('clarification_question'):
#         print("\n--- FINAL RESPONSE (CLARIFICATION) ---"); print(final_state['clarification_question'])
#     else:
#         print("\n--- FINAL SYNTHESIZED RESPONSE ---"); print(final_state['final_response'])

# async def main():
#     await run_agentic_system("Tell me about Alphabet's performance.")
#     await run_agentic_system("Analyze Alphabet's revenue trend for the last two years and discuss how it might relate to the competitive risks mentioned in their latest 10-K.")

# if __name__ == "__main__":
#     try: asyncio.run(main())
#     except (KeyboardInterrupt, asyncio.CancelledError): print("\nExecution interrupted.")


