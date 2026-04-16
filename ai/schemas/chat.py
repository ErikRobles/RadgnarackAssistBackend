from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class RetrievedChunkResponse(BaseModel):
    score: float
    product_name: str
    chunk_type: str
    chunk_content: str
    product_url: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    used_context: bool
    escalation_needed: bool
    status: str
    retrieved_chunks: list[RetrievedChunkResponse]