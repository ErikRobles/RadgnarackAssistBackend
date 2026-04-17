#!/usr/bin/env python3
"""
Embed manual chunks and upsert to Pinecone.
Appends to existing index.
"""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "radgnarack-assist")
EMBEDDING_MODEL = "text-embedding-3-small"


def load_manual_chunks(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunk(chunk: dict, client: OpenAI) -> list[float]:
    """Generate embedding for a single chunk."""
    text = f"Product: {chunk['Product Name']}\nChunk Type: {chunk['Chunk Type']}\nContent: {chunk['Chunk Content']}"

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def upsert_manual_chunks(chunks: list[dict]) -> None:
    """Embed and upsert manual chunks to Pinecone."""
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your-pinecone-api-key-here":
        raise ValueError("PINECONE_API_KEY not set")

    client = OpenAI()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    vectors = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}: {chunk['Chunk Type']}...")

        embedding = embed_chunk(chunk, client)

        # Create unique ID for manual chunks
        vector_id = f"manual-{i:04d}"

        vector = {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "product_name": chunk["Product Name"],
                "chunk_type": chunk["Chunk Type"],
                "chunk_content": chunk["Chunk Content"],
                "product_url": chunk["product URL"],
                "source_type": "manual",
                "document_name": "installation-manual.pdf",
                "topic": "installation",
            }
        }
        vectors.append(vector)
        time.sleep(0.2)  # Rate limiting

    print(f"\nUpserting {len(vectors)} manual chunks to Pinecone...")
    index.upsert(vectors=vectors)

    # Verify
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")
    print(f"Total vectors in index: {stats.total_vector_count}")
    print("\nManual chunks upserted successfully!")


def main():
    chunks_path = "ai/data/manual_chunks.json"

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Manual chunks not found: {chunks_path}")

    chunks = load_manual_chunks(chunks_path)
    print(f"Loaded {len(chunks)} manual chunks")

    upsert_manual_chunks(chunks)


if __name__ == "__main__":
    main()
