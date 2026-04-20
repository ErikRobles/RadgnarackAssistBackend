import asyncio
import unittest
from unittest.mock import patch

from app.api.v1 import telegram as telegram_api
from app.repositories.escalation_repository import escalation_repo
from app.schemas.escalation import EscalationStatus
from app.services import escalation_service


class TelegramRoutingTests(unittest.TestCase):
    def setUp(self):
        escalation_repo._escalations.clear()
        escalation_repo._telegram_msg_index.clear()
        escalation_repo._counter = 0

    def _create_sent_escalation(self, conversation_id="conv-test", message_id=1001):
        escalation = escalation_repo.create(
            user_question="What color racks do you offer?",
            conversation_id=conversation_id,
        )
        escalation_repo.update_status(escalation.escalation_id, EscalationStatus.SENT_TO_OWNER)
        escalation_repo.update_telegram_message_id(
            escalation.escalation_id,
            telegram_message_id=message_id,
            telegram_chat_id="owner-chat",
        )
        return escalation

    def test_reply_to_message_path_still_records_owner_reply(self):
        escalation = self._create_sent_escalation(message_id=2001)
        payload = {
            "message": {
                "text": "The racks come in black and white.",
                "chat": {"id": "owner-chat"},
                "reply_to_message": {"message_id": 2001, "text": "original escalation"},
            }
        }

        with patch.object(escalation_service, "process_learning", return_value=None):
            result = asyncio.run(telegram_api.telegram_webhook(payload))

        updated = escalation_repo.get(escalation.escalation_id)
        self.assertTrue(result["ok"])
        self.assertEqual(updated.status, EscalationStatus.OWNER_REPLIED)
        self.assertEqual(updated.owner_reply, "The racks come in black and white.")

    def test_reply_command_records_owner_reply_end_to_end(self):
        escalation = self._create_sent_escalation()
        payload = {
            "message": {
                "text": f"/reply {escalation.escalation_id} The racks come in black and white.",
                "chat": {"id": "owner-chat"},
            }
        }

        with patch.object(escalation_service, "process_learning", return_value=None):
            result = asyncio.run(telegram_api.telegram_webhook(payload))

        updated = escalation_repo.get(escalation.escalation_id)
        self.assertTrue(result["ok"])
        self.assertEqual(updated.status, EscalationStatus.OWNER_REPLIED)
        self.assertEqual(updated.owner_reply, "The racks come in black and white.")

    def test_plain_text_escalation_id_fallback_records_cleaned_reply(self):
        escalation = self._create_sent_escalation()
        payload = {
            "message": {
                "text": f"For {escalation.escalation_id}: Generally the racks come in black and white.",
                "chat": {"id": "owner-chat"},
            }
        }

        with patch.object(escalation_service, "process_learning", return_value=None):
            result = asyncio.run(telegram_api.telegram_webhook(payload))

        updated = escalation_repo.get(escalation.escalation_id)
        self.assertTrue(result["ok"])
        self.assertEqual(updated.status, EscalationStatus.OWNER_REPLIED)
        self.assertEqual(updated.owner_reply, "Generally the racks come in black and white.")

    def test_plain_text_escalation_id_fallback_is_case_insensitive(self):
        escalation = self._create_sent_escalation()
        payload = {
            "message": {
                "text": f"{escalation.escalation_id.upper()} Black and white are standard colors.",
                "chat": {"id": "owner-chat"},
            }
        }

        with patch.object(escalation_service, "process_learning", return_value=None):
            result = asyncio.run(telegram_api.telegram_webhook(payload))

        updated = escalation_repo.get(escalation.escalation_id)
        self.assertTrue(result["ok"])
        self.assertEqual(updated.status, EscalationStatus.OWNER_REPLIED)
        self.assertEqual(updated.owner_reply, "Black and white are standard colors.")

    def test_unresolved_plain_text_message_does_not_attach_to_escalation(self):
        escalation = self._create_sent_escalation()
        payload = {
            "message": {
                "text": "Black and white are standard colors.",
                "chat": {"id": "owner-chat"},
            }
        }

        with patch.object(escalation_service, "process_learning", return_value=None):
            result = asyncio.run(telegram_api.telegram_webhook(payload))

        unchanged = escalation_repo.get(escalation.escalation_id)
        self.assertTrue(result["ok"])
        self.assertEqual(unchanged.status, EscalationStatus.SENT_TO_OWNER)
        self.assertIsNone(unchanged.owner_reply)


if __name__ == "__main__":
    unittest.main()
