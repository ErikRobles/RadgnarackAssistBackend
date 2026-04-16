import hashlib
import re
from typing import List, Dict, Any, Optional
from app.adapters import embeddings, pinecone

async def ingest_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ingestion Pipeline: Cleaning -> Chunking -> Embedding -> Upsert.
    Strict failure behavior: Malformed doc is skipped; adapter failure stops run.
    """
    summary = {
        "documents_processed": 0,
        "documents_skipped": 0,
        "chunks_created": 0,
        "upserts_attempted": 0,
        "errors": []
    }
    
    for doc in documents:
        try:
            # 1. Basic Validation
            if not doc.get("content") or not doc.get("type"):
                raise ValueError("Missing required fields: content or type.")
                
            # 2. Derive stable document_id
            doc_id = doc.get("document_id")
            if not doc_id:
                doc_id = _derive_doc_id(doc.get("source", "unknown"))
            
            # 3. Semantic Chunking
            chunks = _chunk_document(doc["content"])
            
            # 4. Generate Embeddings (Full run fail on error)
            vectors = await embeddings.embed_texts(chunks)
            
            # 5. Shape Vector Payloads
            upsert_payload = []
            for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                vector_id = f"{doc_id}_{str(i).zfill(3)}"
                upsert_payload.append({
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "text": chunk_text,
                        "type": doc["type"],
                        "source": doc.get("source", "unknown"),
                        "chunk_index": i,
                        "document_id": doc_id
                    }
                })
            
            # 6. Pinecone Upsert
            await pinecone.upsert_vectors(upsert_payload)
            
            summary["documents_processed"] += 1
            summary["chunks_created"] += len(chunks)
            summary["upserts_attempted"] += len(chunks)
            
        except ValueError as e:
            summary["documents_skipped"] += 1
            summary["errors"].append(f"Doc {doc.get('source', 'unknown')}: {str(e)}")
            continue
            
    return summary

def _derive_doc_id(source: str) -> str:
    """Normalize source to a stable, URL-safe identifier."""
    clean = re.sub(r"\.[^.]+$", "", source).lower()
    clean = re.sub(r"[^a-z0-9]+", "-", clean).strip("-")
    return clean or hashlib.md5(source.encode()).hexdigest()[:8]

def _chunk_document(content: str, token_limit: int = 300) -> List[str]:
    """
    Deterministic semantic-section chunking.
    Splits by double newlines or headers.
    """
    # Simple semantic splitting for Phase 1
    # Split by double newlines OR markdown headers
    # We use a lookahead to keep headers with their content
    raw_sections = re.split(r"\n\n+|(?=^#{1,3}\s)", content, flags=re.MULTILINE)
    
    sections = [s.strip() for s in raw_sections if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for section in sections:
        # Approximation: 4 chars per token
        section_length = len(section) // 4
        
        if current_length + section_length > token_limit and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(section)
        current_length += section_length
        
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks
