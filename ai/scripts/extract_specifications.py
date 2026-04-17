#!/usr/bin/env python3
"""
Extract product specifications from PDF and prepare for embedding.
Preserves page-level provenance and conflicting values as separate chunks.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Optional
from pypdf import PdfReader


@dataclass
class SpecificationChunk:
    product_name: str
    section_title: str
    page_number: int
    chunk_text: str
    chunk_index: int
    source_file: str
    document_type: str
    topic: str
    ingest_version: str = "1.0"


def clean_text(text: str) -> str:
    """Remove footer/nav noise from extracted text."""
    # Remove common footer/nav patterns
    patterns_to_remove = [
        r'SHOP\s+NO\s+W\s+HOME.*?SHOP',  # Navigation links
        r'4/\d+/\d+,?\s*\d+:\d+\s*(?:AM|PM)?',  # Date/timestamp
        r'Product Speci[ﬁf]ications',  # Page title repeated
        r'https://radgnarack\.com/product-specifications',  # URL
        r'Phone:\s*\(\d{3}\)\s*\d{3}-\d{4}',  # Phone
        r'Operating hours:.*?MST',  # Hours
        r'Useful Links.*?Copyright © \d{4} RadGnaRack\.',  # Footer block
        r'Shop\s+About\s+Us\s+Contact\s+Support\s+Blog\s+Events\s+Warranty',  # Footer links
        r'Social\s+Media.*?Instagram',  # Social block
    ]
    
    cleaned = text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up whitespace
    cleaned = re.sub(r'\n+', '\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    return cleaned.strip()


def parse_specifications(text: str) -> dict:
    """Parse specification lines into structured data."""
    specs = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or '–' not in line:
            continue
        
        # Split on en-dash or hyphen
        if '–' in line:
            parts = line.split('–', 1)
        elif '-' in line:
            parts = line.split('-', 1)
        else:
            continue
        
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            specs[key] = value
    
    return specs


def extract_product_name(text: str) -> Optional[str]:
    """Extract product name from page text."""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.isupper() and len(line) > 10 and not line.startswith('HTTP'):
            # Likely product name
            if any(word in line for word in ['MODULAR', 'ATTACHMENT', 'BIKE', 'RACK']):
                return line
    return None


def extract_all_product_sections(text: str) -> list[tuple[str, str]]:
    """
    Extract all product sections from page text.
    Returns list of (product_name, section_text) tuples.
    Handles multiple products on same page.
    """
    lines = text.split('\n')
    sections = []
    current_product = None
    current_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if this line is a product heading
        is_product_heading = (
            line_stripped.isupper() 
            and len(line_stripped) > 10 
            and not line_stripped.startswith('HTTP')
            and any(word in line_stripped for word in ['MODULAR', 'ATTACHMENT', 'BIKE', 'RACK'])
        )
        
        if is_product_heading:
            # Save previous section if exists
            if current_product and current_lines:
                section_text = '\n'.join(current_lines).strip()
                if section_text:
                    sections.append((current_product, section_text))
            # Start new section
            current_product = line_stripped
            current_lines = []
        elif current_product is not None:
            # Add line to current section
            current_lines.append(line)
    
    # Don't forget the last section
    if current_product and current_lines:
        section_text = '\n'.join(current_lines).strip()
        if section_text:
            sections.append((current_product, section_text))
    
    return sections


def create_normalized_text(product_name: str, specs: dict, page_number: int = 0) -> str:
    """Create normalized retrieval-friendly text with semantic grouping for dimensions."""
    lines = [
        f"Product: {product_name}",
        f"Document Type: specification",
    ]
    
    if page_number > 0:
        lines.append(f"Page: {page_number}")
    
    # Add Weight first if present
    if 'Weight' in specs:
        lines.append(f"Weight: {specs['Weight']}")
    
    # Add Dimensions grouping label when Length/Width/Height are present
    has_length = 'Length' in specs
    has_width = 'Width' in specs
    has_height = 'Height' in specs
    
    if has_length or has_width or has_height:
        lines.append("")
        lines.append("Dimensions (Length, Width, Height):")
        if has_length:
            lines.append(f"Length: {specs['Length']}")
        if has_width:
            lines.append(f"Width: {specs['Width']}")
        if has_height:
            lines.append(f"Height: {specs['Height']}")
    
    # Add Load Capacity
    if 'Load Capacity' in specs:
        lines.append("")
        lines.append(f"Load Capacity: {specs['Load Capacity']}")
    
    # Add Bike Load Capacity fields
    if 'Bike Load Capacity Front' in specs or 'Bike Load Capacity Back' in specs:
        lines.append("")
        if 'Bike Load Capacity Front' in specs:
            lines.append(f"Bike Load Capacity Front: {specs['Bike Load Capacity Front']}")
        if 'Bike Load Capacity Back' in specs:
            lines.append(f"Bike Load Capacity Back: {specs['Bike Load Capacity Back']}")
    
    # Add any remaining specs not in standard groups
    standard_keys = {'Weight', 'Length', 'Width', 'Height', 'Load Capacity', 
                     'Bike Load Capacity Front', 'Bike Load Capacity Back'}
    remaining = {k: v for k, v in specs.items() if k not in standard_keys}
    if remaining:
        lines.append("")
        for key, value in remaining.items():
            lines.append(f"{key}: {value}")
    
    return '\n'.join(lines)


def extract_chunks_from_pdf(pdf_path: str) -> List[SpecificationChunk]:
    """Extract specification chunks from PDF with page-level provenance.
    Handles multiple products on the same page."""
    reader = PdfReader(pdf_path)
    chunks = []
    chunk_index = 0
    
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        if not raw_text:
            continue
        
        # Skip footer pages
        if page_num == 5:
            continue
        
        # Clean text
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        
        # Extract all product sections from this page
        product_sections = extract_all_product_sections(cleaned)
        
        if not product_sections:
            # Fallback to single-product extraction
            product_name = extract_product_name(cleaned)
            if not product_name:
                continue
            product_sections = [(product_name, cleaned)]
        
        # Process each product section
        for product_name, section_text in product_sections:
            # Parse specifications for this section
            specs = parse_specifications(section_text)
            if not specs:
                continue
            
            # Create normalized text
            normalized_text = create_normalized_text(product_name, specs, page_num)
            
            # Determine topic based on product
            if 'HIGH MODULAR' in product_name:
                topic = 'high-modular-attachment-bar'
            elif 'LOW MODULAR' in product_name:
                topic = 'low-modular-attachment-bar'
            elif 'ALL BIKE' in product_name:
                topic = 'all-bike-rack'
            else:
                topic = 'product-specification'
            
            chunk = SpecificationChunk(
                product_name=product_name,
                section_title='Product Specifications',
                page_number=page_num,
                chunk_text=normalized_text,
                chunk_index=chunk_index,
                source_file='ProductSpecifications.pdf',
                document_type='specification',
                topic=topic,
            )
            
            chunks.append(chunk)
            chunk_index += 1
    
    return chunks


def save_chunks_to_json(chunks: List[SpecificationChunk], output_path: str) -> None:
    """Save chunks to JSON format compatible with embedding pipeline."""
    records = []
    for chunk in chunks:
        records.append({
            "Product Name": chunk.product_name,
            "Chunk Type": chunk.section_title,
            "Chunk Content": chunk.chunk_text,
            "product URL": "https://radgnarack.com",
            "source_file": chunk.source_file,
            "document_type": chunk.document_type,
            "section_title": chunk.section_title,
            "product_name": chunk.product_name,
            "page_number": chunk.page_number,
            "topic": chunk.topic,
            "chunk_index": chunk.chunk_index,
            "ingest_version": chunk.ingest_version,
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(records)} specification chunks to {output_path}")
    
    # Print summary
    for chunk in chunks:
        print(f"  Page {chunk.page_number}: {chunk.product_name}")


def main():
    pdf_path = "ai/data/specifications/ProductSpecifications.pdf"
    output_path = "ai/data/specification_chunks.json"
    
    print(f"Extracting specifications from {pdf_path}...")
    chunks = extract_chunks_from_pdf(pdf_path)
    
    print(f"\nExtracted {len(chunks)} chunks:")
    save_chunks_to_json(chunks, output_path)
    
    print("\n=== CONFLICT DETECTION ===")
    high_modular_long = [c for c in chunks if 'HIGH MODULAR ATTACHMENT BAR - LONG' in c.product_name]
    if len(high_modular_long) > 1:
        print(f"Found {len(high_modular_long)} versions of HIGH MODULAR ATTACHMENT BAR - LONG:")
        for chunk in high_modular_long:
            print(f"  Page {chunk.page_number}: {chunk.chunk_text[:100]}...")
    
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
