import os
import json
import asyncio
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_dicts

# --- 1. Define the Structured Metadata Model (from article) ---
class ChunkMetadata(BaseModel):
    """Structured metadata for a document chunk."""
    summary: str = Field(description="A concise 1-2 sentence summary of the chunk.")
    keywords: List[str] = Field(description="A list of 5-7 key topics or entities mentioned.")
    hypothetical_questions: List[str] = Field(description="A list of 3-5 questions this chunk could answer.")
    table_summary: Optional[str] = Field(description="If the chunk is a table, a natural language summary of its key insights.", default=None)

# --- 2. LLM Enrichment Logic (Adapted for Gemini API) ---

def get_enrichment_schema() -> Dict[str, Any]:
    """Generates the JSON schema for the Gemini API call from the Pydantic model."""
    schema = ChunkMetadata.schema()
    return {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
            "hypothetical_questions": {"type": "ARRAY", "items": {"type": "STRING"}},
            "table_summary": {"type": "STRING"},
        },
        "required": ["summary", "keywords", "hypothetical_questions"]
    }

def generate_enrichment_prompt(chunk_text: str, is_table: bool) -> str:
    """Generates a prompt for the LLM to enrich a chunk."""
    table_instruction = "This chunk is a TABLE. Your summary should describe the main data points and trends, for example: 'This table shows a 15% year-over-year increase in revenue for the Cloud segment.'" if is_table else ""

    return f"""
    You are an expert financial analyst. Please analyze the following document chunk and generate the specified metadata in a valid JSON format.
    {table_instruction}
    Chunk Content:
    ---
    {chunk_text}
    ---
    """

async def enrich_chunk(chunk) -> Optional[Dict[str, Any]]:
    """Enriches a single chunk with LLM-generated metadata using Gemini."""
    is_table = 'text_as_html' in chunk.metadata.to_dict()
    content = chunk.metadata.text_as_html if is_table else chunk.text
    truncated_content = content[:4000]  # Truncate long chunks to fit context window

    prompt = generate_enrichment_prompt(truncated_content, is_table)
    
    apiKey = "AIzaSyBbsYPL_IG4lPkAlXL6AG0Q0SEi1fbVnHs" 
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": get_enrichment_schema()
        }
    }

    try:
        # NOTE: In a real-world scenario, you'd use an async HTTP client like aiohttp
        # For simplicity here, we'll use a basic fetch-like structure (conceptual).
        # This part requires an actual async HTTP request implementation.
        # Let's simulate the API call for now. A real implementation would be needed.
        # A more robust solution would involve retries and proper session management.
        
        # This is a conceptual placeholder for a proper async fetch call.
        # To make this runnable, you would replace this section with a library like `aiohttp`.
        # For example:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload) as response:
        #         result = await response.json()

        # Let's mock a successful response structure for demonstration
        # In a real run, this would be the actual API call.
        
        # This is a placeholder for the actual async fetch call
        # In a real environment, you'd use a library like `httpx` or `aiohttp`
        # Since we can't make external calls directly here, we'll structure the code
        # as if we were. A user would need to run this in an environment with network access.
        
        # A simple, direct fetch simulation is not possible in all environments.
        # The logic below assumes a successful call and parsing.
        # print(f"  - Simulating enrichment for a {'table' if is_table else 'text'} chunk...")
        # For now, let's just return a placeholder. The logic below shows how you would process a real response.
        # return None
        
        # The following code would be used with a real async HTTP library
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()
            result = response.json()

        if result.get('candidates'):
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_text)
        return None

    except Exception as e:
        print(f"  - Error enriching chunk: {e}")
        return None

# --- 3. Main Processing Logic ---

async def main():
    """Main function to parse, chunk, enrich, and save the data."""
    ENRICHED_CHUNKS_PATH = 'data/enriched_chunks.json'
    SUBMISSION_FILE = 'data/alphabet_10k_submission.txt'

    if os.path.exists(ENRICHED_CHUNKS_PATH):
        print(f"Found existing enriched chunks file. Loading from disk: '{ENRICHED_CHUNKS_PATH}'")
        with open(ENRICHED_CHUNKS_PATH, 'r') as f:
            all_enriched_chunks = json.load(f)
    else:
        print("No existing enriched chunks file found. Starting full processing.")
        
        if not os.path.exists(SUBMISSION_FILE):
             print(f"FATAL: Submission file not found at '{SUBMISSION_FILE}'. Please run download script.")
             return

        parsed_dicts = partition_html(filename=SUBMISSION_FILE, infer_table_structure=True, strategy='fast')
        elements = elements_from_dicts([el.to_dict() for el in parsed_dicts])
        doc_chunks = chunk_by_title(elements, max_characters=2048, combine_text_under_n_chars=256)
        
        all_enriched_chunks = []
        
        # Use tqdm.asyncio for an async-compatible progress bar
        async for chunk in tqdm(doc_chunks, desc="Enriching Chunks"):
            enrichment_data = await enrich_chunk(chunk)
            if enrichment_data:
                is_table = 'text_as_html' in chunk.metadata.to_dict()
                content = chunk.metadata.text_as_html if is_table else chunk.text
                
                final_chunk_data = {
                    'source': "10-K/alphabet_10k_submission.txt",
                    'content': content,
                    'is_table': is_table,
                    **enrichment_data
                }
                all_enriched_chunks.append(final_chunk_data)

        print(f"\nCompleted processing. Total enriched chunks: {len(all_enriched_chunks)}")

        os.makedirs('data', exist_ok=True)
        with open(ENRICHED_CHUNKS_PATH, 'w') as f:
            json.dump(all_enriched_chunks, f, indent=2)
        print(f"Enriched chunks saved to '{ENRICHED_CHUNKS_PATH}'.")

    # Display a sample of the enriched data
    if all_enriched_chunks:
        print("\n--- Sample of Enriched Data ---")
        sample_data = all_enriched_chunks[5] # Show a sample chunk
        print(json.dumps(sample_data, indent=2))


if __name__ == "__main__":
    # To run this, you'll need httpx: pip install httpx
    # Then you can run the main async function
    asyncio.run(main())
