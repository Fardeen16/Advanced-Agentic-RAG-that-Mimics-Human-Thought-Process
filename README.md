# Advanced Agentic RAG that Mimics Human Thought Process
</br>
This project implements a sophisticated, multi-agent RAG (Retrieval-Augmented Generation) system designed to perform complex financial analysis. Inspired by the advanced techniques detailed in the article "Building an Advanced Agentic RAG Pipeline", this system goes beyond simple Q&A to mimic a human analyst's thought process, featuring cognitive loops for planning, self-correction, and insight generation.

The agent can intelligently query both unstructured documents (like SEC 10-K filings) and structured data (financial metrics), access real-time information from the web, and synthesize its findings into a coherent, insightful narrative.

Project Overview
The core of this project is an agentic system built with LangChain and LangGraph. It employs a "Supervisor" model that manages a team of specialized AI "analysts." When a user submits a query, the system doesn't just find an answer; it creates a multi-step plan, executes it, audits the results of each step, and can even re-plan if its initial approach fails. This enables it to handle complex, multi-part questions that would cause a standard RAG pipeline to fail.

The project is broken down into three main phases:

Data Foundation: Preparing and enriching both unstructured and structured financial data.

Specialist Agents: Building a team of specialized tools, each with a unique analytical skill.

The Agentic Supervisor: Constructing the cognitive architecture that allows the agent to plan, reason, and self-correct.

Phase 1: Data Foundation
Before any analysis can be performed, the agent needs a reliable knowledge base. This phase involves preparing two distinct types of data sources.

1.1 Unstructured Data Preparation
Source: Publicly available SEC 10-K annual reports (e.g., for Alphabet Inc.).

Process: We programmatically download the full HTML submission of the latest 10-K filing. The raw HTML is then parsed using the unstructured library to intelligently partition the document into logical elements like titles, narrative text, and tables.

Enrichment: Each chunk is then "enriched" using an LLM to generate high-quality metadata, including a concise summary, keywords, and a list of hypothetical questions the chunk could answer. This metadata is crucial for enabling powerful semantic search later on.

1.2 Structured Data Preparation
Source: A CSV file containing historical, key financial metrics (e.g., quarterly revenue and net income).

Process: This data is loaded into a persistent SQLite database, providing the agent with a reliable source for factual, point-in-time lookups.

Phase 2: Building the Specialist Agents
The Supervisor manages a team of four specialist tools, each designed for a specific task:

The Librarian (librarian_rag_tool): An expert in deep document retrieval. It uses an advanced three-step process (Query Optimization -> Vector Search -> Cross-Encoder Re-ranking) to find the most relevant text and table chunks from the enriched 10-K filings.

The SQL Analyst (analyst_sql_tool): A quantitative expert. This tool connects to the SQLite database and uses a LangChain SQL Agent to answer specific questions about financial metrics (e.g., "What was the total revenue in 2023?").

The Trend Analyst (analyst_trend_tool): A time-series specialist. It queries the entire financial history from the SQLite database to calculate and summarize trends, such as Quarter-over-Quarter (QoQ) and Year-over-Year (YoY) growth.

The Scout (scout_web_search_tool): The team's connection to the real world. It uses DuckDuckGo Search to find live, up-to-the-minute information that isn't in the static documents, such as current stock prices or breaking news.

Phase 3: The Agentic Supervisor
This is the cognitive core of the project, implemented as a state machine using LangGraph. The Supervisor guides the agent through a sophisticated reasoning loop:

The Gatekeeper: The first node in our graph. It inspects the user's query for ambiguity. If the query is too vague, it generates a clarifying question and halts. This prevents the agent from wasting resources on poorly defined tasks.

The Planner: Once a query is approved, the Planner creates a multi-step plan by selecting the appropriate tools in the correct sequence to answer the user's request.

The Tool Executor: Executes the plan step-by-step, calling the designated specialist tools.

The Auditor: After each tool runs, this self-correction node reviews the output. It uses an LLM to score the result for relevance and consistency against the user's original query.

The Conditional Router: The central decision-maker. Based on the Auditor's score, the router decides what to do next. If the score is high, it continues the plan. If the score is low, it triggers a re-planning loop, sending feedback to the Planner to create a new, better strategy.

The Strategist (Synthesizer): Once all steps are successfully completed, this final node synthesizes the findings from all tool outputs into a single, comprehensive answer. Crucially, it is instructed to infer potential causal links between data points (e.g., connecting a financial trend to a documented risk) and present them as a data-grounded hypothesis.
