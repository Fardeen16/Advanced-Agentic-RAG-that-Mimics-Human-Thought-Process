import json
import sqlite3
import pandas as pd
import qdrant_client
import httpx
from typing import List, Dict, Any

# --- Configuration ---
ENRICHED_CHUNKS_PATH = 'data/enriched_chunks.json'
STRUCTURED_DATA_PATH = 'data/alphabet_financials_structured.csv'
DB_PATH = "data/financials.db"
TABLE_NAME = "financials_summary"
COLLECTION_NAME = "financial_docs_alphabet"

# --- 1. Populate Vector Store (Qdrant) ---

def create_embedding_text(chunk: Dict) -> str:
    """Creates a combined text string for embedding from an enriched chunk."""
    return f"""
    Summary: {chunk['summary']}
    Keywords: {', '.join(chunk['keywords'])}
    Content: {chunk['content'][:1000]}
    """

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a batch of texts using Gemini API."""
    apiKey = "AIzaSyBbsYPL_IG4lPkAlXL6AG0Q0SEi1fbVnHs"
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={apiKey}"
    
    requests = [{"model": "models/text-embedding-004", "content": {"parts": [{"text": t}]}} for t in texts]
    payload = {"requests": requests}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()
            result = response.json()
            return [embedding['values'] for embedding in result.get('embeddings', [])]
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error during embedding: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"An error occurred during embedding: {e}")
    return []


async def populate_vector_store():
    """Loads enriched data, generates embeddings, and upserts into Qdrant."""
    print("--- 1. Populating Vector Store (Qdrant) ---")
    
    try:
        with open(ENRICHED_CHUNKS_PATH, 'r') as f:
            all_enriched_chunks = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Enriched chunks file not found at '{ENRICHED_CHUNKS_PATH}'. Please run the enrichment script first.")
        return

    # Using an in-memory Qdrant instance as in the article
    client = qdrant_client.QdrantClient(path="data/qdrant_db")

    
    # Get embedding dimension from a sample call
    print("Fetching embedding dimension...")
    sample_embedding = await generate_embeddings_batch(["sample text"])
    if not sample_embedding:
        print("FATAL: Could not generate a sample embedding. Aborting.")
        return
    embedding_dimension = len(sample_embedding[0])
    print(f"Embedding dimension: {embedding_dimension}")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_client.http.models.VectorParams(
            size=embedding_dimension,
            distance=qdrant_client.http.models.Distance.COSINE
        )
    )
    print(f"Qdrant collection '{COLLECTION_NAME}' created.")

    points_to_upsert = []
    texts_to_embed = [create_embedding_text(chunk) for chunk in all_enriched_chunks]

    print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
    # Process in batches
    batch_size = 100 # Gemini batchEmbedContents API has a limit of 100
    all_embeddings = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_embeddings = await generate_embeddings_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)
        print(f"  - Embedded batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")

    if len(all_embeddings) != len(all_enriched_chunks):
        print("FATAL: Mismatch between number of chunks and generated embeddings. Aborting upsert.")
        return

    for i, (chunk, embedding) in enumerate(zip(all_enriched_chunks, all_embeddings)):
        points_to_upsert.append(qdrant_client.http.models.PointStruct(
            id=i,
            payload=chunk,
            vector=embedding
        ))

    print(f"Prepared {len(points_to_upsert)} points. Upserting into Qdrant...")
    # --- FIX: Removed the unsupported 'batch_size' argument ---
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points_to_upsert,
        wait=True
    )
    print("Upsert complete!")
    
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Points in collection: {collection_info.points_count}")
    return client # Return client for potential use later

# --- 2. Populate Relational Store (SQLite) ---

def populate_relational_store():
    """Loads structured data from CSV into a SQLite database."""
    print("\n--- 2. Populating Relational Store (SQLite) ---")
    
    try:
        df = pd.read_csv(STRUCTURED_DATA_PATH)
    except FileNotFoundError:
        print(f"FATAL: Structured data file not found at '{STRUCTURED_DATA_PATH}'. Please run the download script.")
        return None

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()
    print(f"SQLite database created/updated at '{DB_PATH}'.")

    # Use LangChain's wrapper as in the article for verification
    from langchain_community.utilities import SQLDatabase
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    
    print("\nVerifying table schema:")
    print(db.get_table_info())
    
    print("\nVerifying sample rows:")
    print(db.run(f"SELECT * FROM {TABLE_NAME} LIMIT 5"))
    return db # Return db object for use later


async def main():
    """Main function to run both population steps."""
    await populate_vector_store()
    populate_relational_store()


if __name__ == "__main__":
    import asyncio
    # To run this, you'll need httpx: pip install httpx
    asyncio.run(main())