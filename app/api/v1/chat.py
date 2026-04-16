from fastapi import APIRouter, HTTPException
import os

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.escalation_service import create_escalation, should_escalate

# Use Pinecone if API key is configured, otherwise fall back to local JSON
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY and PINECONE_API_KEY != "your-pinecone-api-key-here":
    from ai.services.pinecone_rag_service import answer_question, result_to_dict
else:
    from ai.services.rag_service import answer_question, result_to_dict

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(question)
    result_dict = result_to_dict(result)

    # Check if escalation is needed
    if should_escalate(result_dict.get("status"), result_dict.get("escalation_needed")):
        # Create escalation asynchronously (don't block response)
        create_escalation(
            user_question=question,
            conversation_id=request.conversation_id if hasattr(request, "conversation_id") else None,
            page_context=request.page_context if hasattr(request, "page_context") else None,
            source_url=request.source_url if hasattr(request, "source_url") else None,
        )

    return ChatResponse(**result_dict)