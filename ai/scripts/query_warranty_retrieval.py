#!/usr/bin/env python3
"""Validate warranty retrieval against the existing Pinecone index."""
import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "radgnarack-assist")
EMBEDDING_MODEL = "text-embedding-3-small"

QUERIES = [
    "What does the warranty cover?",
    "Does improper installation void the warranty?",
    "Is damage from misuse covered?",
    "How do I make a warranty claim?",
    "What is excluded from the warranty?",
]


def main() -> None:
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your-pinecone-api-key-here":
        raise ValueError("PINECONE_API_KEY not set")

    client = OpenAI()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    for query in QUERIES:
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
        result = index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True,
            filter={"document_type": {"$eq": "warranty"}},
        )
        print(f"\n=== QUERY: {query} ===")
        for rank, match in enumerate(result.matches, start=1):
            metadata = match.metadata or {}
            content = metadata.get("chunk_content", "")
            preview = content[:700].replace("\n", " ")
            print(
                f"{rank}. score={match.score:.4f} "
                f"section={metadata.get('section_title')} "
                f"page={metadata.get('page_number')} "
                f"topic={metadata.get('topic')}"
            )
            print(f"   {preview}")


if __name__ == "__main__":
    main()
