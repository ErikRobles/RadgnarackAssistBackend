import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class EmbeddedChunk:
    product_name: str
    chunk_type: str
    chunk_content: str
    product_url: str
    embedding_model: str
    embedding: list[float]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_embedding_input(product_name: str, chunk_type: str, chunk_content: str) -> str:
    return (
        f"Product: {product_name}\n"
        f"Chunk Type: {chunk_type}\n"
        f"Content: {chunk_content}"
    )


def read_chunks_from_excel(filepath: str, sheet_name: str = "CLEAN_CHUNKS") -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    required_columns = ["Product Name", "Chunk Type", "Chunk Content", "product URL"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required_columns:
        df[col] = df[col].apply(normalize_text)

    df = df[
        (df["Product Name"] != "")
        & (df["Chunk Type"] != "")
        & (df["Chunk Content"] != "")
        ].copy()

    return df


def embed_chunks(
        df: pd.DataFrame,
        model: str = "text-embedding-3-small",
        batch_size: int = 50,
        sleep_between_batches: float = 0.5,
) -> list[EmbeddedChunk]:
    client = OpenAI()
    embedded_rows: list[EmbeddedChunk] = []

    texts: list[str] = []
    row_metadata: list[dict[str, str]] = []

    for _, row in df.iterrows():
        product_name = row["Product Name"]
        chunk_type = row["Chunk Type"]
        chunk_content = row["Chunk Content"]
        product_url = row["product URL"]

        texts.append(build_embedding_input(product_name, chunk_type, chunk_content))
        row_metadata.append(
            {
                "product_name": product_name,
                "chunk_type": chunk_type,
                "chunk_content": chunk_content,
                "product_url": product_url,
            }
        )

    total = len(texts)
    print(f"Preparing to embed {total} chunks using model: {model}")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_meta = row_metadata[start:end]

        print(f"Embedding rows {start + 1} to {end} of {total}...")

        response = client.embeddings.create(
            model=model,
            input=batch_texts,
        )

        for meta, item in zip(batch_meta, response.data):
            embedded_rows.append(
                EmbeddedChunk(
                    product_name=meta["product_name"],
                    chunk_type=meta["chunk_type"],
                    chunk_content=meta["chunk_content"],
                    product_url=meta["product_url"],
                    embedding_model=model,
                    embedding=item.embedding,
                )
            )

        time.sleep(sleep_between_batches)

    return embedded_rows


def save_embeddings_to_json(records: list[EmbeddedChunk], output_path: str) -> None:
    serializable = [asdict(record) for record in records]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} embedded chunks to {output_path}")


def main() -> None:
    excel_path = "ai/data/radgnarack_chunks.xlsx"
    sheet_name = "CLEAN_CHUNKS"
    output_path = "ai/data/radgnarack_embeddings.json"
    model = "text-embedding-3-small"

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    df = read_chunks_from_excel(excel_path, sheet_name=sheet_name)
    embedded_records = embed_chunks(df, model=model, batch_size=50)
    save_embeddings_to_json(embedded_records, output_path)


if __name__ == "__main__":
    main()