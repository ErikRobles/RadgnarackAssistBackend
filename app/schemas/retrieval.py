from pydantic import BaseModel
from typing import List, Dict, Any

class KnowledgeChunk(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

class RetrievalResult(BaseModel):
    documents: List[KnowledgeChunk]
    query_metadata: Dict[str, Any]
    is_sufficient: bool
