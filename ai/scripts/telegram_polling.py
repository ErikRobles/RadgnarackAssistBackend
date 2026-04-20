#!/usr/bin/env python3
"""
Telegram bot polling script for local development.
Polls Telegram API for new messages and processes replies.
Run this in a separate terminal during development.
"""
import os
import sys
import time
import re

import requests

sys.path.insert(0, '/Users/ejames/dev/RadgnarackAssist/RadgnarackAssistBackend')

from app.adapters.telegram import telegram_adapter
from app.services.escalation_service import process_owner_reply
from app.repositories.escalation_repository import escalation_repo

TELEGRAM_API_BASE = "https://api.telegram.org/bot"
POLL_INTERVAL = 5  # seconds
ESCALATION_ID_PATTERN = re.compile(r"\b(esc_[A-Za-z0-9_]+)\b", re.IGNORECASE)


def extract_plain_text_escalation_reply(text: str) -> tuple[str, str] | None:
    """Find an escalation id anywhere in plain text and return cleaned reply text."""
    match = ESCALATION_ID_PATTERN.search(text or "")
    if not match:
        return None

    escalation_id = match.group(1).lower()
    prefix = text[:match.start()].strip()
    suffix = text[match.end():]
    if prefix.lower() in {"", "for", "re", "regarding"}:
        reply_text = re.sub(r"^[\s:：,;\-–—]+", "", suffix).strip()
    else:
        reply_text = (text[:match.start()] + text[match.end():]).strip()
        reply_text = re.sub(r"\s{2,}", " ", reply_text)

    if not reply_text:
        return None
    return escalation_id, reply_text


def poll_telegram_updates():
    """Poll Telegram for new messages and process owner replies."""
    if not telegram_adapter.enabled:
        print("Telegram not configured. Exiting.")
        return

    bot_token = telegram_adapter.bot_token
    last_update_id = 0

    print("Starting Telegram polling...")
    print(f"Polling interval: {POLL_INTERVAL}s")
    print("Press Ctrl+C to stop")
    print()

    while True:
        try:
            # Get updates from Telegram
            url = f"{TELEGRAM_API_BASE}{bot_token}/getUpdates"
            params = {
                "offset": last_update_id + 1,
                "limit": 100,
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("ok"):
                print(f"Telegram API error: {data}")
                time.sleep(POLL_INTERVAL)
                continue

            updates = data.get("result", [])

            for update in updates:
                update_id = update.get("update_id")
                if update_id > last_update_id:
                    last_update_id = update_id

                message = update.get("message", {})
                if not message:
                    continue

                text = message.get("text", "").strip()
                chat_id = str(message.get("chat", {}).get("id", ""))

                # Skip if not from owner
                if chat_id != telegram_adapter.owner_chat_id:
                    continue

                if not text:
                    continue

                print(f"Received message: {text[:50]}...")

                # Try reply_to_message first
                reply_to = message.get("reply_to_message")
                if reply_to:
                    original_message_id = reply_to.get("message_id")
                    original_text = reply_to.get("text", "")

                    print(f"  Reply to message {original_message_id}")

                    # Look up escalation by message_id
                    escalation = escalation_repo.get_by_telegram_message_id(
                        chat_id=chat_id,
                        message_id=original_message_id
                    )

                    if escalation:
                        print(f"  Found escalation: {escalation.escalation_id}")
                        updated = process_owner_reply(escalation.escalation_id, text)
                        if updated:
                            print(f"  ✓ Reply recorded for {escalation.escalation_id}")
                            # Acknowledge receipt
                            send_acknowledgment(bot_token, chat_id, escalation.escalation_id)
                        continue

                    # Fallback: parse escalation ID from original message text
                    import re
                    match = re.search(r"\*ID:\* `([^`]+)`", original_text)
                    if match:
                        escalation_id = match.group(1)
                        print(f"  Parsed escalation ID from text: {escalation_id}")
                        updated = process_owner_reply(escalation_id, text)
                        if updated:
                            print(f"  ✓ Reply recorded for {escalation_id}")
                            send_acknowledgment(bot_token, chat_id, escalation_id)
                        continue

                # Try /reply command fallback
                reply_data = telegram_adapter.parse_reply_command(text)
                if reply_data:
                    escalation_id, reply_text = reply_data
                    print(f"  /reply parsed: {escalation_id}")
                    escalation = escalation_repo.get(escalation_id)
                    if not escalation:
                        print(f"  ✗ /reply escalation lookup failure: {escalation_id}")
                        continue
                    print(f"  ✓ /reply escalation lookup success: {escalation_id}")
                    updated = process_owner_reply(escalation_id, reply_text)
                    if updated:
                        print(f"  ✓ /reply process_owner_reply success: {escalation_id}")
                        send_acknowledgment(bot_token, chat_id, escalation_id)
                    else:
                        print(f"  ✗ /reply process_owner_reply failure: {escalation_id}")
                    continue

                # Last fallback: plain text containing an escalation id.
                plain_reply = extract_plain_text_escalation_reply(text)
                if plain_reply:
                    escalation_id, reply_text = plain_reply
                    print(f"  Plain-text escalation id parsed: {escalation_id}")
                    escalation = escalation_repo.get(escalation_id)
                    if not escalation:
                        print(f"  ✗ Plain-text escalation lookup failure: {escalation_id}")
                        continue
                    print(f"  ✓ Plain-text escalation lookup success: {escalation_id}")
                    updated = process_owner_reply(escalation_id, reply_text)
                    if updated:
                        print(f"  ✓ Plain-text process_owner_reply success: {escalation_id}")
                        send_acknowledgment(bot_token, chat_id, escalation_id)
                    else:
                        print(f"  ✗ Plain-text process_owner_reply failure: {escalation_id}")

            if updates:
                print(f"Processed {len(updates)} updates")

        except KeyboardInterrupt:
            print("\nStopping polling...")
            break
        except Exception as e:
            print(f"Error during polling: {e}")

        time.sleep(POLL_INTERVAL)


def send_acknowledgment(bot_token: str, chat_id: str, escalation_id: str):
    """Send acknowledgment that reply was received."""
    try:
        url = f"{TELEGRAM_API_BASE}{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"✓ Reply recorded for escalation {escalation_id}",
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send acknowledgment: {e}")


if __name__ == "__main__":
    poll_telegram_updates()
