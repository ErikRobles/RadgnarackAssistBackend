from typing import List, Dict, Any, Optional
from app.schemas.retrieval import RetrievalResult, KnowledgeChunk
from app.adapters import pinecone, llm_provider
from app.core.config import settings

async def retrieve_faq_context(
    query: str, 
    filters: Optional[Dict[str, Any]] = None, 
    top_k: Optional[int] = None
) -> RetrievalResult:
    """Coordinates retrieval of business knowledge for FAQ and grounding."""
    
    # 1. Configuration resolution
    active_top_k = top_k or settings.DEFAULT_TOP_K
    
    # 2. Embedding generation boundary
    query_vector = await llm_provider.embed_query(query)
    
    # 3. Vector query via adapter
    raw_results = await pinecone.query_index(
        vector=query_vector, 
        filters=filters, 
        top_k=active_top_k
    )
    
    # 4. Result shaping and threshold filtering
    documents: List[KnowledgeChunk] = []
    is_sufficient = False
    
    for item in raw_results:
        # Check if item passes the configurable threshold
        score = item.get("score", 0.0)
        metadata = item.get("metadata", {})
        
        # In this phase, we map raw adapter result to KnowledgeChunk
        chunk = KnowledgeChunk(
            text=metadata.get("text", ""),
            score=score,
            metadata=metadata
        )
        
        documents.append(chunk)
        
        # Mark as sufficient if at least one document passes the threshold
        if score >= settings.MIN_RELEVANCE_SCORE:
            is_sufficient = True
            
    return RetrievalResult(
        documents=documents,
        query_metadata={
            "top_k": active_top_k,
            "filters": filters
        },
        is_sufficient=is_sufficient
    )
