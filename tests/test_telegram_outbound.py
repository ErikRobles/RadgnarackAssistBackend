import unittest
from unittest.mock import patch

import requests

from app.adapters.telegram import TelegramAdapter


class FakeSuccessResponse:
    status_code = 200
    text = '{"ok": true}'

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "ok": True,
            "result": {
                "message_id": 1234,
                "chat": {"id": "owner-chat"},
            },
        }


class FakeBadRequestResponse:
    status_code = 400
    text = "Bad Request: can't parse entities"

    def raise_for_status(self):
        raise requests.HTTPError("400 Client Error: Bad Request", response=self)

    def json(self):
        return {"ok": False, "description": self.text}


class TelegramOutboundTests(unittest.TestCase):
    def _adapter(self):
        adapter = TelegramAdapter()
        adapter.bot_token = "test-token"
        adapter.owner_chat_id = "owner-chat"
        adapter.enabled = True
        return adapter

    def test_escalation_send_payload_has_no_parse_mode_and_contains_fallback_instruction(self):
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeSuccessResponse()

        adapter = self._adapter()
        with patch("app.adapters.telegram.requests.post", side_effect=fake_post):
            result = adapter.send_escalation(
                escalation_id="esc_00001",
                user_question="What color racks do you offer?",
                conversation_id="conv-test",
            )

        self.assertEqual(result["telegram_message_id"], 1234)
        self.assertEqual(captured["json"]["chat_id"], "owner-chat")
        self.assertNotIn("parse_mode", captured["json"])
        self.assertIn(
            "If Telegram reply mode fails, send: /reply esc_00001 Your answer here",
            captured["json"]["text"],
        )

    def test_bad_request_failure_log_includes_status_body_chat_id_and_parse_mode(self):
        adapter = self._adapter()

        with patch("app.adapters.telegram.requests.post", return_value=FakeBadRequestResponse()):
            with self.assertLogs("app.adapters.telegram", level="ERROR") as logs:
                result = adapter.send_escalation(
                    escalation_id="esc_00001",
                    user_question="What color racks do you offer?",
                )

        self.assertIsNone(result)
        output = "\n".join(logs.output)
        self.assertIn("status=400", output)
        self.assertIn("Bad Request: can't parse entities", output)
        self.assertIn("chat_id=owner-chat", output)
        self.assertIn("parse_mode=None", output)


if __name__ == "__main__":
    unittest.main()
