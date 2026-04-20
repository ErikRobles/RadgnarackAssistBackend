"""
Telegram webhook endpoint for receiving owner replies.
"""
import logging
import re
from fastapi import APIRouter, HTTPException, status

from app.adapters.telegram import telegram_adapter
from app.services.escalation_service import process_owner_reply

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/telegram", tags=["telegram"])

ESCALATION_ID_PATTERN = re.compile(r"\b(esc_[A-Za-z0-9_]+)\b", re.IGNORECASE)


class TelegramWebhookPayload:
    """Expected Telegram webhook payload structure."""
    update_id: int
    message: dict


def _extract_plain_text_escalation_reply(text: str) -> tuple[str, str] | None:
    """Find an escalation id anywhere in plain text and return cleaned reply text."""
    match = ESCALATION_ID_PATTERN.search(text or "")
    if not match:
        return None

    escalation_id = match.group(1).lower()
    prefix = text[:match.start()].strip()
    suffix = text[match.end():]

    # If the id is being used as a routing prefix, remove it and common separators.
    if prefix.lower() in {"", "for", "re", "regarding"}:
        reply_text = re.sub(r"^[\s:：,;\-–—]+", "", suffix).strip()
    else:
        reply_text = (text[:match.start()] + text[match.end():]).strip()
        reply_text = re.sub(r"\s{2,}", " ", reply_text)

    if not reply_text:
        return None
    return escalation_id, reply_text


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def telegram_webhook(payload: dict):
    """
    Receive Telegram webhook updates.
    Process owner replies via reply_to_message or /reply command.
    """
    from app.repositories.escalation_repository import escalation_repo

    # DEBUG: Log full inbound payload
    logger.warning(f"[WEBHOOK] Full payload: {payload}")

    # Extract message from payload
    message = payload.get("message", {})
    if not message:
        return {"ok": True}

    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))

    # DEBUG: Log extracted fields
    logger.warning(f"[WEBHOOK] text={repr(text)}, chat_id={chat_id}")

    if not text:
        logger.warning("[WEBHOOK] Empty text, returning ok")
        return {"ok": True}

    # PRIMARY PATH: Check if this is a reply to an escalation message
    reply_to = message.get("reply_to_message")
    logger.warning(f"[WEBHOOK] reply_to_message present: {reply_to is not None}")
    if reply_to:
        original_message_id = reply_to.get("message_id")
        logger.warning(f"[WEBHOOK] original_message_id (raw): {original_message_id}")
        if original_message_id:
            # Convert to int since Telegram webhook sends numbers as strings in JSON
            # but we store them as integers for consistent lookup
            original_message_id_int = int(original_message_id)
            logger.warning(f"[WEBHOOK] original_message_id (converted): {original_message_id_int}")
            # Look up escalation by Telegram message_id
            escalation = escalation_repo.get_by_telegram_message_id(
                chat_id=chat_id,
                message_id=original_message_id_int
            )
            logger.warning(f"[WEBHOOK] Escalation lookup result: {escalation}")
            if escalation:
                logger.warning(f"[WEBHOOK] Found escalation: {escalation.escalation_id}, calling process_owner_reply")
                updated = process_owner_reply(escalation.escalation_id, text)
                logger.warning(f"[WEBHOOK] process_owner_reply result: {updated}")
                if updated:
                    logger.warning(f"[WEBHOOK] SUCCESS: Reply recorded for {escalation.escalation_id}")
                    return {
                        "ok": True,
                        "message": f"Reply recorded for escalation {escalation.escalation_id}"
                    }
                else:
                    logger.warning(f"[WEBHOOK] process_owner_reply returned None")
            else:
                logger.warning(f"[WEBHOOK] No escalation found for message_id {original_message_id_int}")
        # Fallback: check if the original message text contains an escalation ID
        original_text = reply_to.get("text", "")
        match = re.search(r"\*ID:\* `([^`]+)`", original_text)
        if match:
            escalation_id = match.group(1)
            logger.warning(f"[WEBHOOK] reply_to text escalation_id parsed: {escalation_id}")
            updated = process_owner_reply(escalation_id, text)
            logger.warning(f"[WEBHOOK] reply_to text process_owner_reply result: {updated}")
            if updated:
                return {
                    "ok": True,
                    "message": f"Reply recorded for escalation {escalation_id}"
                }

    # FALLBACK PATH: Check if this is a /reply command
    logger.warning(f"[WEBHOOK] Checking for /reply command in: {repr(text)}")
    reply_data = telegram_adapter.parse_reply_command(text)
    logger.warning(f"[WEBHOOK] parse_reply_command result: {reply_data}")
    if reply_data:
        escalation_id, reply_text = reply_data
        logger.warning(f"[WEBHOOK] /reply parsed escalation_id={escalation_id}")
        escalation = escalation_repo.get(escalation_id)
        if not escalation:
            logger.warning(f"[WEBHOOK] /reply escalation lookup failure escalation_id={escalation_id}")
            return {
                "ok": False,
                "message": f"Escalation {escalation_id} not found"
            }

        logger.warning(f"[WEBHOOK] /reply escalation lookup success escalation_id={escalation_id}")
        updated = process_owner_reply(escalation_id, reply_text)
        logger.warning(f"[WEBHOOK] /reply process_owner_reply result: {updated}")
        if updated:
            logger.warning(f"[WEBHOOK] /reply process_owner_reply success escalation_id={escalation_id}")
            return {
                "ok": True,
                "message": f"Reply recorded for escalation {escalation_id}"
            }

        logger.warning(f"[WEBHOOK] /reply process_owner_reply failure escalation_id={escalation_id}")
        return {
            "ok": False,
            "message": f"Reply was not recorded for escalation {escalation_id}"
        }

    # LAST FALLBACK PATH: Plain text containing an escalation id.
    if not reply_to:
        plain_reply = _extract_plain_text_escalation_reply(text)
        logger.warning(f"[WEBHOOK] plain-text escalation fallback result: {plain_reply}")
        if plain_reply:
            escalation_id, reply_text = plain_reply
            escalation = escalation_repo.get(escalation_id)
            if not escalation:
                logger.warning(f"[WEBHOOK] plain-text escalation lookup failure escalation_id={escalation_id}")
                return {"ok": True, "message": "No matching escalation found"}

            logger.warning(f"[WEBHOOK] plain-text escalation lookup success escalation_id={escalation_id}")
            updated = process_owner_reply(escalation_id, reply_text)
            logger.warning(f"[WEBHOOK] plain-text process_owner_reply result: {updated}")
            if updated:
                logger.warning(f"[WEBHOOK] plain-text process_owner_reply success escalation_id={escalation_id}")
                return {
                    "ok": True,
                    "message": f"Reply recorded for escalation {escalation_id}"
                }
            logger.warning(f"[WEBHOOK] plain-text process_owner_reply failure escalation_id={escalation_id}")

    logger.warning("[WEBHOOK] No escalation routing resolved for inbound owner message")
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
