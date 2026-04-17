"""
In-memory repository for escalations.
In production, this should be replaced with a persistent database.
"""
from datetime import datetime
from typing import Optional

from app.schemas.escalation import Escalation, EscalationStatus


class EscalationRepository:
    def __init__(self):
        # In-memory storage for escalations
        self._escalations: dict[str, Escalation] = {}
        self._counter: int = 0
        # Index for Telegram message_id lookup
        self._telegram_msg_index: dict[tuple[str, int], str] = {}

    def _generate_id(self) -> str:
        """Generate a unique escalation ID."""
        self._counter += 1
        return f"esc_{self._counter:05d}"

    def create(
        self,
        user_question: str,
        conversation_id: Optional[str] = None,
        page_context: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> Escalation:
        """Create a new escalation record."""
        escalation_id = self._generate_id()
        escalation = Escalation(
            escalation_id=escalation_id,
            conversation_id=conversation_id,
            user_question=user_question,
            page_context=page_context,
            source_url=source_url,
            status=EscalationStatus.PENDING,
            created_at=datetime.utcnow(),
        )
        self._escalations[escalation_id] = escalation
        return escalation

    def get(self, escalation_id: str) -> Optional[Escalation]:
        """Get an escalation by ID."""
        return self._escalations.get(escalation_id)

    def get_by_telegram_message_id(self, chat_id: str, message_id: int) -> Optional[Escalation]:
        """Get an escalation by Telegram message_id."""
        escalation_id = self._telegram_msg_index.get((chat_id, message_id))
        if escalation_id:
            return self._escalations.get(escalation_id)
        return None

    def get_active_by_conversation_id(self, conversation_id: str) -> Optional[Escalation]:
        """Get the latest active escalation for a conversation."""
        active = [
            esc for esc in self._escalations.values()
            if esc.conversation_id == conversation_id
            and esc.status in (EscalationStatus.PENDING, EscalationStatus.SENT_TO_OWNER)
        ]
        if not active:
            return None
        return max(active, key=lambda esc: esc.created_at)

    def close_active_for_conversation(self, conversation_id: str) -> None:
        """Close active escalations for a conversation without changing reply mapping."""
        for esc in self._escalations.values():
            if (
                esc.conversation_id == conversation_id
                and esc.status in (EscalationStatus.PENDING, EscalationStatus.SENT_TO_OWNER)
            ):
                esc.status = EscalationStatus.CLOSED

    def update_telegram_message_id(
        self,
        escalation_id: str,
        telegram_message_id: int,
        telegram_chat_id: str
    ) -> Optional[Escalation]:
        """Update escalation with Telegram message details."""
        escalation = self._escalations.get(escalation_id)
        if escalation:
            escalation.telegram_message_id = telegram_message_id
            escalation.telegram_chat_id = telegram_chat_id
            # Index for lookup
            self._telegram_msg_index[(telegram_chat_id, telegram_message_id)] = escalation_id
        return escalation

    def update_status(
        self,
        escalation_id: str,
        status: EscalationStatus,
        owner_reply: Optional[str] = None,
    ) -> Optional[Escalation]:
        """Update escalation status and optionally add owner reply."""
        escalation = self._escalations.get(escalation_id)
        if escalation:
            escalation.status = status
            if owner_reply:
                escalation.owner_reply = owner_reply
                escalation.owner_replied_at = datetime.utcnow()
        return escalation

    def add_owner_reply(self, escalation_id: str, reply: str) -> Optional[Escalation]:
        """Add owner reply to an escalation."""
        escalation = self._escalations.get(escalation_id)
        if escalation:
            escalation.owner_reply = reply
            escalation.owner_replied_at = datetime.utcnow()
            escalation.status = EscalationStatus.OWNER_REPLIED
        return escalation

    def list_pending(self) -> list[Escalation]:
        """List all pending escalations."""
        return [
            esc for esc in self._escalations.values()
            if esc.status in (EscalationStatus.PENDING, EscalationStatus.SENT_TO_OWNER)
        ]


# Global repository instance
escalation_repo = EscalationRepository()
