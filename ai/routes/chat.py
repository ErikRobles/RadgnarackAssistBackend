from fastapi import APIRouter, HTTPException

from ai.schemas.chat import ChatRequest, ChatResponse
from ai.services.rag_service import answer_question, result_to_dict

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(question)
    result_dict = result_to_dict(result)

    return ChatResponse(**result_dict)