#!/usr/bin/env python3
"""
Embed specification chunks and upsert to Pinecone.
Preserves page-level provenance and conflicting values.
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


def load_spec_chunks(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunk(chunk: dict, client: OpenAI) -> list[float]:
    """Generate embedding for a specification chunk."""
    text = f"Product: {chunk['Product Name']}\nChunk Type: {chunk['Chunk Type']}\nContent: {chunk['Chunk Content']}"
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def upsert_spec_chunks(chunks: list[dict]) -> None:
    """Embed and upsert specification chunks to Pinecone."""
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your-pinecone-api-key-here":
        raise ValueError("PINECONE_API_KEY not set")
    
    client = OpenAI()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    vectors = []
    
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}: {chunk['Product Name']} (Page {chunk['page_number']})...")
        
        embedding = embed_chunk(chunk, client)
        
        # Create unique ID with page number for conflict resolution
        # Format: spec-{product}-{page}-{index}
        product_slug = chunk['product_name'].lower().replace(' ', '-').replace('_', '-')
        vector_id = f"spec-{product_slug}-p{chunk['page_number']}-{i:04d}"
        
        vector = {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "product_name": chunk["Product Name"],
                "chunk_type": chunk["Chunk Type"],
                "chunk_content": chunk["Chunk Content"],
                "product_url": chunk["product URL"],
                "source_file": chunk["source_file"],
                "document_type": chunk["document_type"],
                "section_title": chunk["section_title"],
                "product_name_detail": chunk["product_name"],
                "page_number": chunk["page_number"],
                "topic": chunk["topic"],
                "chunk_index": chunk["chunk_index"],
                "ingest_version": chunk["ingest_version"],
            }
        }
        vectors.append(vector)
        time.sleep(0.2)  # Rate limiting
    
    print(f"\nUpserting {len(vectors)} specification chunks to Pinecone...")
    index.upsert(vectors=vectors)
    
    # Verify
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")
    print(f"Total vectors in index: {stats.total_vector_count}")
    print("\nSpecification chunks upserted successfully!")
    
    # Print conflict preservation notice
    print("\n=== CONFLICT PRESERVATION ===")
    high_mod_long = [v for v in vectors if 'high-modular-attachment-bar-long' in v['id']]
    if len(high_mod_long) > 1:
        print(f"Preserved {len(high_mod_long)} versions of HIGH MODULAR ATTACHMENT BAR - LONG:")
        for v in high_mod_long:
            print(f"  {v['id']} (Page {v['metadata']['page_number']})")


def main():
    chunks_path = "ai/data/specification_chunks.json"
    
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Specification chunks not found: {chunks_path}")
    
    chunks = load_spec_chunks(chunks_path)
    print(f"Loaded {len(chunks)} specification chunks")
    
    upsert_spec_chunks(chunks)


if __name__ == "__main__":
    main()
