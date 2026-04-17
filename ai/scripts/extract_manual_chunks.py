#!/usr/bin/env python3
"""
Extract chunks from installation manual PDF for embedding.
Creates chunks in the same format as radgnarack_chunks.xlsx
"""
import os
import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


@dataclass
class ManualChunk:
    product_name: str
    chunk_type: str
    chunk_content: str
    product_url: str


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from PDF."""
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def split_into_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text into semantic chunks.
    Tries to split on section headers or paragraph boundaries.
    """
    # Normalize whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Try to identify section headers (numbered or uppercase)
    # Pattern: lines starting with numbers or ALL CAPS
    section_pattern = r'(?:^|\n)(?:\d+\.\s+[A-Z]|\d+\.\d+\s+[A-Z]|[A-Z][A-Z\s]{3,}|Step\s+\d+)'

    # Split on section boundaries
    parts = re.split(f'({section_pattern})', text, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""

    for part in parts:
        if not part.strip():
            continue

        if len(current_chunk) + len(part) > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part
        else:
            current_chunk += " " + part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If no semantic chunks found, fall back to paragraph splitting
    if not chunks:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = paragraphs

    return chunks


def create_manual_chunks(pdf_path: str) -> List[ManualChunk]:
    """Extract chunks from installation manual PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Extracting text from {pdf_path}...")
    full_text = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(full_text)} characters")
    print("Splitting into chunks...")

    chunks = split_into_chunks(full_text, max_chunk_size=1200)

    print(f"Created {len(chunks)} chunks")

    manual_chunks = []
    for i, chunk_text in enumerate(chunks, 1):
        manual_chunks.append(ManualChunk(
            product_name="Installation Manual",
            chunk_type="Setup Instructions",
            chunk_content=chunk_text,
            product_url="https://radgnarack.com"
        ))

    return manual_chunks


def save_chunks_to_json(chunks: List[ManualChunk], output_path: str) -> None:
    """Save chunks to JSON format compatible with existing pipeline."""
    import json

    records = []
    for chunk in chunks:
        records.append({
            "Product Name": chunk.product_name,
            "Chunk Type": chunk.chunk_type,
            "Chunk Content": chunk.chunk_content,
            "product URL": chunk.product_url,
            "source_type": "manual",
            "document_name": "installation-manual.pdf",
            "topic": "installation"
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} chunks to {output_path}")


def main():
    pdf_path = "ai/data/manuals/installation-manual.pdf"
    output_path = "ai/data/manual_chunks.json"

    chunks = create_manual_chunks(pdf_path)
    save_chunks_to_json(chunks, output_path)
    print("\nManual extraction complete!")


if __name__ == "__main__":
    main()
