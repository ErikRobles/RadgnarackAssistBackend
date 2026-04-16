import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ---------- CONFIG ----------
EMBEDDINGS_FILE = "ai/data/radgnarack_embeddings.json"
MODEL = "text-embedding-3-small"

# ---------- LOAD DATA ----------
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

client = OpenAI()

# ---------- COSINE SIMILARITY ----------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- QUERY FUNCTION ----------
def query(question, top_k=5):
    print(f"\n🔍 Query: {question}")

    # Embed the question
    response = client.embeddings.create(
        model=MODEL,
        input=question
    )

    query_embedding = response.data[0].embedding

    # Score all chunks
    scored = []
    for item in data:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))

    # Sort by best match
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top results
    results = scored[:top_k]

    print("\nTop matches:\n")
    for score, item in results:
        print(f"Score: {round(score, 4)}")
        print(f"Product: {item['product_name']}")
        print(f"Type: {item['chunk_type']}")
        print(f"Content: {item['chunk_content'][:120]}...")
        print("-" * 50)

# ---------- TEST ----------
if __name__ == "__main__":
    query("Can I carry two eBikes?")
    query("Will this work with a 2 inch hitch?")
    query("How do I secure my bike?")