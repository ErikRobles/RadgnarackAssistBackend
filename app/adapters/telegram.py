"""
Telegram adapter for sending escalation notifications and receiving owner replies.
"""
import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_OWNER_CHAT_ID = os.getenv("TELEGRAM_OWNER_CHAT_ID")
TELEGRAM_API_BASE = "https://api.telegram.org/bot"


class TelegramAdapter:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.owner_chat_id = TELEGRAM_OWNER_CHAT_ID
        self.enabled = bool(self.bot_token and self.owner_chat_id)

    def send_escalation(
        self,
        escalation_id: str,
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
    ) -> Optional[dict]:
        """
        Send escalation notification to owner via Telegram.
        Returns dict with telegram_message_id and telegram_chat_id if successful,
        None if failed.
        """
        if not self.enabled:
            print("Telegram not configured. Skipping notification.")
            return None

        def clean(value: object, limit: int = 700) -> str:
            text = str(value or "").strip()
            text = text.replace("`", "'").replace("*", "")
            return text[:limit] + "..." if len(text) > limit else text

        # Build the message. Keep reply-to behavior unchanged; only enrich text.
        lines = [
            "📋 *New RadgnarackAssist escalation*",
            "",
            f"*ID:* `{escalation_id}`",
        ]

        if conversation_id:
            lines.append(f"*Conversation ID:* `{conversation_id}`")

        lines.extend([
            "",
            "*Original question:*",
            clean(original_question or user_question),
            "",
            "*Latest user message:*",
            clean(latest_user_message or user_question),
        ])

        if conversation_summary:
            lines.extend([
                "",
                "*Conversation summary:*",
                clean(conversation_summary, limit=500),
            ])

        if structured_context:
            context_lines = []
            labels = {
                "vehicle": "Vehicle",
                "vehicle_year": "Vehicle year",
                "bike_type": "Bike type",
                "bike_count": "Bike count",
                "hitch": "Hitch",
            }
            for key, label in labels.items():
                value = structured_context.get(key)
                if value:
                    context_lines.append(f"• {label}: {clean(value, limit=80)}")
            if context_lines:
                lines.extend([
                    "",
                    "*Collected context:*",
                    *context_lines,
                ])

        if escalation_reason:
            lines.extend([
                "",
                "*Escalation reason:*",
                clean(escalation_reason, limit=200),
            ])

        if recent_transcript:
            lines.extend(["", "*Recent transcript:*"])
            for turn in recent_transcript[-8:]:
                role = clean(turn.get("role", ""), limit=20) or "Turn"
                content = clean(turn.get("content", ""), limit=300)
                if content:
                    lines.append(f"{role}: {content}")

        if page_context:
            # Truncate if too long
            context = page_context[:100] + "..." if len(page_context) > 100 else page_context
            lines.append(f"*Context:* {context}")

        if source_url:
            lines.append(f"*Page:* {source_url}")

        lines.extend([
            "",
            "Reply to this message with your answer.",
        ])

        message_text = "\n".join(lines)
        if len(message_text) > 3900:
            message_text = message_text[:3900] + "\n..."

        # Send message via Telegram API
        url = f"{TELEGRAM_API_BASE}{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.owner_chat_id,
            "text": message_text,
            "parse_mode": "Markdown",
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("ok"):
                message_id = result["result"]["message_id"]
                chat_id = str(result["result"]["chat"]["id"])
                print(f"Telegram notification sent for escalation {escalation_id}, message_id: {message_id}")
                return {
                    "telegram_message_id": message_id,
                    "telegram_chat_id": chat_id,
                }
            else:
                print(f"Telegram API returned error: {result}")
                return None
        except requests.RequestException as e:
            print(f"Failed to send Telegram notification: {e}")
            return None

    def parse_reply_command(self, message_text: str) -> Optional[tuple[str, str]]:
        """
        Parse owner reply command.
        Expected format: /reply <escalation_id> <message>
        Returns (escalation_id, reply_text) or None if invalid.
        """
        if not message_text.startswith("/reply"):
            return None

        # Pattern: /reply esc_12345 message text here
        pattern = r"^/reply\s+(esc_\w+)\s+(.+)$"
        match = re.match(pattern, message_text, re.DOTALL)

        if match:
            escalation_id = match.group(1)
            reply_text = match.group(2).strip()
            return escalation_id, reply_text

        return None


# Global adapter instance
telegram_adapter = TelegramAdapter()
