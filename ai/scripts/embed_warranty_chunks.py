#!/usr/bin/env python3
"""
Embed warranty chunks and upsert them to the existing Pinecone index.

This mirrors the existing specification/manual embedding path:
- OpenAI text-embedding-3-small
- existing Pinecone index from PINECONE_INDEX_NAME
- same core metadata keys used by retrieval: product_name, chunk_type,
  chunk_content, product_url
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
CHUNKS_PATH = "ai/data/warranty_chunks.json"


def load_warranty_chunks(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunk(chunk: dict, client: OpenAI) -> list[float]:
    text = (
        f"Product: {chunk['Product Name']}\n"
        f"Chunk Type: {chunk['Chunk Type']}\n"
        f"Document Type: {chunk['document_type']}\n"
        f"Section: {chunk['section_title']}\n"
        f"Topic: {chunk['topic']}\n"
        f"Warranty Topic: {chunk['warranty_topic']}\n"
        f"Content: {chunk['Chunk Content']}"
    )
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def upsert_warranty_chunks(chunks: list[dict]) -> None:
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your-pinecone-api-key-here":
        raise ValueError("PINECONE_API_KEY not set")

    client = OpenAI()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    vectors = []
    for i, chunk in enumerate(chunks):
        print(
            f"Embedding warranty chunk {i + 1}/{len(chunks)}: "
            f"{chunk['section_title']} (Page {chunk['page_number']})..."
        )
        embedding = embed_chunk(chunk, client)
        vector_id = f"warranty-{chunk['ingest_version']}-{chunk['chunk_index']:04d}"
        vectors.append({
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
                "warranty_topic": chunk["warranty_topic"],
                "legal_section_type": chunk["legal_section_type"],
                "embedding_model": EMBEDDING_MODEL,
            },
        })
        time.sleep(0.2)

    print(f"\nUpserting {len(vectors)} warranty chunks to Pinecone index '{PINECONE_INDEX_NAME}'...")
    index.upsert(vectors=vectors)

    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")
    print(f"Total vectors in index: {stats.total_vector_count}")
    print("\nWarranty chunks upserted successfully!")


def main() -> None:
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Warranty chunks not found: {CHUNKS_PATH}")
    chunks = load_warranty_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} warranty chunks")
    upsert_warranty_chunks(chunks)


if __name__ == "__main__":
    main()
