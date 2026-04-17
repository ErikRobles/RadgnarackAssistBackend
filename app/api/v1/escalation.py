"""
Escalation endpoints for checking owner reply status and retrieving replies.
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.repositories.escalation_repository import escalation_repo
from app.schemas.escalation import EscalationStatus

router = APIRouter(prefix="/api/escalation", tags=["escalation"])


@router.get("/check/{conversation_id}")
def check_escalation_status(conversation_id: str):
    """
    Check escalation status for a conversation.
    Returns owner reply if available, or indicates pending status.
    """
    # Find latest active escalation by conversation_id.
    escalation = escalation_repo.get_active_by_conversation_id(conversation_id)
    
    if not escalation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No escalation found for this conversation"
        )
    
    # Check if owner has replied
    if escalation.status == EscalationStatus.OWNER_REPLIED and escalation.owner_reply:
        return {
            "has_reply": True,
            "escalation_id": escalation.escalation_id,
            "status": escalation.status,
            "owner_reply": escalation.owner_reply,
            "owner_replied_at": escalation.owner_replied_at,
            "user_question": escalation.user_question,
        }
    
    # No reply yet
    return {
        "has_reply": False,
        "escalation_id": escalation.escalation_id,
        "status": escalation.status,
        "message": "Waiting for owner response",
    }


@router.get("/reply/{escalation_id}")
def get_escalation_reply(escalation_id: str):
    """
    Get the owner reply for a specific escalation.
    """
    escalation = escalation_repo.get(escalation_id)
    
    if not escalation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Escalation not found"
        )
    
    return {
        "escalation_id": escalation.escalation_id,
        "status": escalation.status,
        "has_reply": escalation.status == EscalationStatus.OWNER_REPLIED,
        "owner_reply": escalation.owner_reply,
        "owner_replied_at": escalation.owner_replied_at,
    }
