#!/usr/bin/env python3
"""
Extract semantic warranty chunks from Warranty.pdf.

Text-first extraction:
- preserves page numbers
- preserves section headings
- removes obvious website nav/footer clutter
- keeps source wording intact inside section bodies
"""
import json
import re
from dataclasses import dataclass
from typing import Iterable

from pypdf import PdfReader


PDF_PATH = "ai/data/warranty/Warranty.pdf"
OUTPUT_PATH = "ai/data/warranty_chunks.json"
INGEST_VERSION = "warranty-1.0"


@dataclass
class WarrantyChunk:
    section_title: str
    page_number: int
    topic: str
    warranty_topic: str
    legal_section_type: str
    chunk_text: str
    chunk_index: int


SECTION_METADATA = {
    "LIMITED LIFETIME WARRANTY": ("coverage-overview", "coverage", "warranty_grant"),
    "WARRANTY COVERAGE": ("coverage-overview", "coverage", "warranty_grant"),
    "WARRANTY DETAILS": ("warranty-details", "coverage", "warranty_terms"),
    "WHAT IS COVERED": ("coverage-overview", "covered-defects", "warranty_terms"),
    "WHAT IS NOT COVERED": ("exclusions", "exclusions", "exclusions"),
    "PROOF OF PURCHASE": ("product-conditions-eligibility", "eligibility", "eligibility"),
    "WARRANTY CLAIMS": ("claims-contact-support-procedure", "claims", "claims_process"),
    "WARRANTY LIMITATIONS": ("warranty-limitations", "limitations", "limitations"),
    "DISCLAIMER": ("limitations-of-liability", "disclaimer", "disclaimer"),
    "LIMITATION OF LIABILITY": ("limitations-of-liability", "liability", "liability_limitation"),
    "YOUR LEGAL RIGHTS": ("legal-rights", "legal-rights", "legal_rights"),
    "EFFECTIVE DATE": ("warranty-period", "duration-coverage-period", "effective_date"),
}


def clean_page_text(text: str) -> str:
    """Remove obvious website navigation/footer clutter while preserving warranty wording."""
    clutter_patterns = [
        r"SHOP\s+NO\s*W",
        r"HOMEHOME",
        r"EVENT\s+SEVENT\s+S",
        r"ABOUTABOUT",
        r"BL\s+OGBL\s+OG",
        r"SHOPSHOP",
        r"\d{1,2}/\d{1,2}/\d{2},?\s+\d{1,2}:\d{2}\s+(?:AM|PM)\s+Warranty",
        r"https://radgnarack\.com/warranty\s+\d+/\d+",
        r"Useful Links\s+Shop\s+About Us\s+Contact\s+Events\s+Warranty\s+Product Speci[ﬁf]ications\s+Social Media\s+Facebook\s+Youtube\s+Instagram",
        r"Support\s+Blog\s+Radgnarack Manual\s+Media Kit\s+Copyright © \d{4} RadGnaRack\.",
        r"CLICK TO DOWNLOAD WARRANTY",
        r"Phone:\s*\(\d{3}\)\s*\d{3}-\d{4}",
        r"Operating hours:\s*9:00 to 5:00 MST",
    ]

    cleaned = text.replace("ﬁ", "fi").replace("ﬀ", "ff")
    for pattern in clutter_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_page_text(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    return pages


def iter_sections(page_text: str) -> Iterable[tuple[str, str]]:
    """Yield (heading, body) from uppercase warranty headings."""
    heading_pattern = re.compile(
        r"^(LIMITED LIFETIME WARRANTY|WARRANTY COVERAGE|WARRANTY DETAILS|WARRANTY CLAIMS|WARRANTY LIMITATIONS|DISCLAIMER|LIMITATION OF LIABILITY|YOUR LEGAL RIGHTS|EFFECTIVE DATE):?$",
        flags=re.MULTILINE,
    )
    matches = list(heading_pattern.finditer(page_text))
    for i, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        body = page_text[start:end].strip(":\n ")
        if body:
            yield heading, body


def split_warranty_details(page_number: int, body: str, start_index: int) -> list[WarrantyChunk]:
    """Split WARRANTY DETAILS into semantic numbered sub-sections."""
    detail_pattern = re.compile(
        r"(?:^|\n)(\d+\.\s+(WHAT IS COVERED|WHAT IS NOT COVERED|PROOF OF PURCHASE):)",
        flags=re.MULTILINE,
    )
    matches = list(detail_pattern.finditer(body))
    chunks: list[WarrantyChunk] = []

    for i, match in enumerate(matches):
        section_title = match.group(2)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_body = body[start:end].strip()
        topic, warranty_topic, legal_section_type = SECTION_METADATA[section_title]
        chunk_text = normalize_chunk_text(
            section_title=section_title.title(),
            page_number=page_number,
            body=section_body,
            topic=topic,
            warranty_topic=warranty_topic,
        )
        chunks.append(WarrantyChunk(
            section_title=section_title.title(),
            page_number=page_number,
            topic=topic,
            warranty_topic=warranty_topic,
            legal_section_type=legal_section_type,
            chunk_text=chunk_text,
            chunk_index=start_index + len(chunks),
        ))

    return chunks


def normalize_chunk_text(
    section_title: str,
    page_number: int,
    body: str,
    topic: str,
    warranty_topic: str,
) -> str:
    lines = [
        f"Section: {section_title}",
        "Document Type: warranty",
        f"Page: {page_number}",
        f"Topic: {topic}",
        f"Warranty Topic: {warranty_topic}",
        "",
        body.strip(),
    ]
    return "\n".join(lines).strip()


def create_chunks(pdf_path: str) -> list[WarrantyChunk]:
    chunks: list[WarrantyChunk] = []

    for page_number, page_text in extract_pages(pdf_path):
        for heading, body in iter_sections(page_text):
            if heading == "LIMITED LIFETIME WARRANTY":
                continue

            if heading == "WARRANTY DETAILS":
                chunks.extend(split_warranty_details(page_number, body, len(chunks)))
                continue

            topic, warranty_topic, legal_section_type = SECTION_METADATA[heading]
            section_title = heading.title()
            chunk_text = normalize_chunk_text(
                section_title=section_title,
                page_number=page_number,
                body=body,
                topic=topic,
                warranty_topic=warranty_topic,
            )
            chunks.append(WarrantyChunk(
                section_title=section_title,
                page_number=page_number,
                topic=topic,
                warranty_topic=warranty_topic,
                legal_section_type=legal_section_type,
                chunk_text=chunk_text,
                chunk_index=len(chunks),
            ))

    return chunks


def save_chunks(chunks: list[WarrantyChunk], output_path: str) -> None:
    records = []
    for chunk in chunks:
        records.append({
            "Product Name": "RadGnaRack Warranty",
            "Chunk Type": chunk.section_title,
            "Chunk Content": chunk.chunk_text,
            "product URL": "https://radgnarack.com/warranty",
            "source_file": "Warranty.pdf",
            "document_type": "warranty",
            "section_title": chunk.section_title,
            "product_name": "RadGnaRack Warranty",
            "page_number": chunk.page_number,
            "topic": chunk.topic,
            "chunk_index": chunk.chunk_index,
            "ingest_version": INGEST_VERSION,
            "warranty_topic": chunk.warranty_topic,
            "legal_section_type": chunk.legal_section_type,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} warranty chunks to {output_path}")
    for record in records:
        print(
            f"  chunk {record['chunk_index']}: page {record['page_number']} "
            f"{record['section_title']} ({record['topic']})"
        )


def main() -> None:
    chunks = create_chunks(PDF_PATH)
    save_chunks(chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()
