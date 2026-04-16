#!/usr/bin/env python3
"""
Initialize Pinecone index and upsert all chunks from the local embeddings file.
Run this script once to populate Pinecone, or re-run when chunks change.
"""
import hashlib
import json
import os
import re

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "radgnarack-assist")
EMBEDDINGS_FILE = "ai/data/radgnarack_embeddings.json"

DIMENSION = 1536  # text-embedding-3-small dimension


def generate_stable_id(product_name: str, chunk_type: str, index: int) -> str:
    """Generate a stable deterministic ID for a chunk."""
    # Normalize for ID generation
    normalized_product = re.sub(r"[^a-zA-Z0-9_-]", "-", product_name.lower())
    normalized_type = re.sub(r"[^a-zA-Z0-9_-]", "-", chunk_type.lower())
    base = f"{normalized_product}--{normalized_type}--{index}"
    # Hash to ensure uniqueness and valid ID format
    hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
    return f"chunk-{hash_suffix}"


def load_chunks(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def init_pinecone_index(pc: Pinecone, index_name: str) -> None:
    """Initialize Pinecone index if it doesn't exist."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")


def prepare_vectors(chunks: list[dict]) -> list[dict]:
    """Prepare vectors for Pinecone upsert with stable IDs and metadata."""
    vectors = []

    for i, chunk in enumerate(chunks):
        # Generate stable ID
        vector_id = generate_stable_id(
            chunk["product_name"],
            chunk["chunk_type"],
            i
        )

        # Prepare metadata
        metadata = {
            "product_name": chunk["product_name"],
            "chunk_type": chunk["chunk_type"],
            "chunk_content": chunk["chunk_content"],
            "product_url": chunk["product_url"],
            "embedding_model": chunk.get("embedding_model", "text-embedding-3-small"),
        }

        # Preserve optional fields if present
        if "is_faq" in chunk:
            metadata["is_faq"] = chunk["is_faq"]
        if "source_doc" in chunk:
            metadata["source_doc"] = chunk["source_doc"]
        if "source_sheet" in chunk:
            metadata["source_sheet"] = chunk["source_sheet"]

        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": metadata,
        })

    return vectors


def upsert_in_batches(index, vectors: list[dict], batch_size: int = 100) -> None:
    """Upsert vectors to Pinecone in batches."""
    total = len(vectors)
    for i in range(0, total, batch_size):
        batch = vectors[i : i + batch_size]
        print(f"Upserting batch {i // batch_size + 1}/{(total - 1) // batch_size + 1} ({len(batch)} vectors)...")
        index.upsert(vectors=batch)
    print(f"Upserted {total} vectors total.")


def main():
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your-pinecone-api-key-here":
        raise ValueError("PINECONE_API_KEY is not set in environment.")

    print("Loading chunks from local file...")
    chunks = load_chunks(EMBEDDINGS_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    print("\nInitializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    init_pinecone_index(pc, PINECONE_INDEX_NAME)

    index = pc.Index(PINECONE_INDEX_NAME)

    print("\nPreparing vectors...")
    vectors = prepare_vectors(chunks)

    print("\nUpserting to Pinecone...")
    upsert_in_batches(index, vectors)

    # Verify
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")
    print(f"Total vectors in index: {stats.total_vector_count}")

    print("\nDone! Pinecone is ready for queries.")


if __name__ == "__main__":
    main()
