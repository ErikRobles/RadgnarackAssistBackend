"""
Escalation service for handling unanswered questions.
Creates escalation records and sends notifications to owner.
"""
from typing import Optional

from app.adapters.telegram import telegram_adapter
from app.repositories.escalation_repository import escalation_repo
from app.schemas.escalation import Escalation, EscalationStatus


def create_escalation(
    user_question: str,
    conversation_id: Optional[str] = None,
    page_context: Optional[str] = None,
    source_url: Optional[str] = None,
    original_question: Optional[str] = None,
    latest_user_message: Optional[str] = None,
    conversation_summary: Optional[str] = None,
    structured_context: Optional[dict] = None,
    escalation_reason: Optional[str] = None,
    recent_transcript: Optional[list[dict[str, str]]] = None,
) -> Optional[Escalation]:
    """
    Create an escalation record and notify owner.
    Returns the created escalation or None if creation failed.
    """
    # Create escalation record
    escalation = escalation_repo.create(
        user_question=user_question,
        conversation_id=conversation_id,
        page_context=page_context,
        source_url=source_url,
    )

    # Send Telegram notification
    telegram_result = telegram_adapter.send_escalation(
        escalation_id=escalation.escalation_id,
        user_question=user_question,
        conversation_id=conversation_id,
        page_context=page_context,
        source_url=source_url,
        original_question=original_question,
        latest_user_message=latest_user_message,
        conversation_summary=conversation_summary,
        structured_context=structured_context,
        escalation_reason=escalation_reason,
        recent_transcript=recent_transcript,
    )

    # Update status and store Telegram message details
    if telegram_result:
        escalation_repo.update_status(
            escalation_id=escalation.escalation_id,
            status=EscalationStatus.SENT_TO_OWNER,
        )
        escalation_repo.update_telegram_message_id(
            escalation_id=escalation.escalation_id,
            telegram_message_id=telegram_result["telegram_message_id"],
            telegram_chat_id=telegram_result["telegram_chat_id"],
        )
        escalation.status = EscalationStatus.SENT_TO_OWNER
        escalation.telegram_message_id = telegram_result["telegram_message_id"]
        escalation.telegram_chat_id = telegram_result["telegram_chat_id"]

    return escalation


def process_owner_reply(escalation_id: str, reply_text: str) -> Optional[Escalation]:
    """
    Process owner reply received via Telegram.
    Returns updated escalation or None if not found.
    """
    escalation = escalation_repo.get(escalation_id)
    if not escalation:
        return None

    updated = escalation_repo.add_owner_reply(escalation_id, reply_text)
    return updated


def get_escalation(escalation_id: str) -> Optional[Escalation]:
    """Get an escalation by ID."""
    return escalation_repo.get(escalation_id)


def should_escalate(status: str, escalation_needed: bool) -> bool:
    """
    Determine if a chat response should trigger escalation.
    Only escalate when status is insufficient_context AND escalation_needed is true.
    """
    return status == "insufficient_context" and escalation_needed
