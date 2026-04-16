from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EscalationStatus(str, Enum):
    PENDING = "pending"
    SENT_TO_OWNER = "sent_to_owner"
    OWNER_REPLIED = "owner_replied"
    CLOSED = "closed"


class Escalation(BaseModel):
    escalation_id: str = Field(..., description="Unique escalation identifier")
    conversation_id: Optional[str] = Field(None, description="Frontend conversation ID if available")
    user_question: str = Field(..., description="The question that triggered escalation")
    page_context: Optional[str] = Field(None, description="Page context if available")
    source_url: Optional[str] = Field(None, description="Page URL if available")
    status: EscalationStatus = Field(default=EscalationStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    owner_reply: Optional[str] = Field(None, description="Owner's response")
    owner_replied_at: Optional[datetime] = Field(None)
    telegram_message_id: Optional[int] = Field(None, description="Telegram message ID for reply matching")
    telegram_chat_id: Optional[str] = Field(None, description="Telegram chat ID")


class EscalationResponse(BaseModel):
    escalation_id: str
    status: EscalationStatus
    message: str
