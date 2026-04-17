"""
Telegram webhook endpoint for receiving owner replies.
"""
from fastapi import APIRouter, HTTPException, status

from app.adapters.telegram import telegram_adapter
from app.services.escalation_service import process_owner_reply

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


class TelegramWebhookPayload:
    """Expected Telegram webhook payload structure."""
    update_id: int
    message: dict


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def telegram_webhook(payload: dict):
    """
    Receive Telegram webhook updates.
    Process owner replies via reply_to_message or /reply command.
    """
    from app.repositories.escalation_repository import escalation_repo

    # Extract message from payload
    message = payload.get("message", {})
    if not message:
        return {"ok": True}

    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))

    if not text:
        return {"ok": True}

    # PRIMARY PATH: Check if this is a reply to an escalation message
    reply_to = message.get("reply_to_message")
    if reply_to:
        original_message_id = reply_to.get("message_id")
        if original_message_id:
            # Convert to int since Telegram webhook sends numbers as strings in JSON
            # but we store them as integers for consistent lookup
            original_message_id = int(original_message_id)
            # Look up escalation by Telegram message_id
            escalation = escalation_repo.get_by_telegram_message_id(
                chat_id=chat_id,
                message_id=original_message_id
            )
            if escalation:
                updated = process_owner_reply(escalation.escalation_id, text)
                if updated:
                    return {
                        "ok": True,
                        "message": f"Reply recorded for escalation {escalation.escalation_id}"
                    }
        # Fallback: check if the original message text contains an escalation ID
        original_text = reply_to.get("text", "")
        import re
        match = re.search(r"\*ID:\* `([^`]+)`", original_text)
        if match:
            escalation_id = match.group(1)
            updated = process_owner_reply(escalation_id, text)
            if updated:
                return {
                    "ok": True,
                    "message": f"Reply recorded for escalation {escalation_id}"
                }

    # FALLBACK PATH: Check if this is a /reply command
    reply_data = telegram_adapter.parse_reply_command(text)
    if reply_data:
        escalation_id, reply_text = reply_data
        updated = process_owner_reply(escalation_id, reply_text)
        if updated:
            return {
                "ok": True,
                "message": f"Reply recorded for escalation {escalation_id}"
            }
        else:
            return {
                "ok": False,
                "message": f"Escalation {escalation_id} not found"
            }

    # Not a reply, acknowledge receipt
    return {"ok": True}


@router.get("/test")
async def test_telegram():
    """Test endpoint to verify Telegram configuration."""
    if telegram_adapter.enabled:
        return {
            "status": "configured",
            "owner_chat_id": telegram_adapter.owner_chat_id,
        }
    else:
        return {
            "status": "not_configured",
            "message": "TELEGRAM_BOT_TOKEN or TELEGRAM_OWNER_CHAT_ID not set"
        }
