import unittest
from datetime import datetime
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from app.services import approved_escalation_learning as learning


def _escalation(reply="Yes — the rack comes in black powder coat finish."):
    return SimpleNamespace(
        escalation_id="esc_00001",
        conversation_id="conv-1",
        user_question="What color does the rack come in?",
        source_url="https://example.com/product",
        created_at=datetime(2026, 4, 20, 12, 0, 0),
        owner_reply=reply,
        owner_replied_at=datetime(2026, 4, 20, 12, 5, 0),
        telegram_chat_id="123",
    )


class ApprovedEscalationLearningTests(unittest.TestCase):
    def test_should_learn_rejects_short_ambiguous_and_nolearn(self):
        self.assertFalse(learning.should_learn("Yes"))
        self.assertFalse(learning.should_learn("NOLEARN: internal note only"))
        self.assertFalse(learning.should_learn("I'm not sure, ask them for pictures"))
        self.assertTrue(learning.should_learn("The rack comes in a black powder coat finish."))

    def test_normalize_escalation_payload_and_hash_are_deterministic(self):
        first = learning.normalize_escalation(_escalation())
        second = learning.normalize_escalation(_escalation())

        self.assertEqual(first["id"], f"approved_qa_{first['content_hash']}")
        self.assertEqual(first["content_hash"], second["content_hash"])
        self.assertEqual(first["question_text"], "What color does the rack come in?")
        self.assertEqual(first["answer_text"], "Yes — the rack comes in black powder coat finish.")
        self.assertEqual(first["topic"], "product_info")
        self.assertEqual(first["source"], "telegram_escalation")
        self.assertEqual(first["approval_state"], "owner_approved")
        self.assertEqual(first["status"], "pending")
        self.assertIn("Approved owner Q&A", first["embedding_text"])
        self.assertIn("Question: What color does the rack come in?", first["embedding_text"])
        self.assertIn("Answer: Yes — the rack comes in black powder coat finish.", first["embedding_text"])

    def test_write_to_ledger_appends_full_payload(self):
        with TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "approved_escalation_qa.jsonl"
            with patch.object(learning, "LEDGER_PATH", ledger):
                payload = learning.normalize_escalation(_escalation())
                learning.write_to_ledger(payload)

            line = ledger.read_text().strip()
            self.assertIn('"question_text": "What color does the rack come in?"', line)
            self.assertIn('"status": "pending"', line)

    def test_process_learning_duplicate_detection_skips_upsert(self):
        with TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "approved_escalation_qa.jsonl"
            with patch.object(learning, "LEDGER_PATH", ledger), \
                 patch.object(learning, "embed_payload", side_effect=AssertionError("should not embed duplicate")), \
                 patch.object(learning, "upsert_to_pinecone", side_effect=AssertionError("should not upsert duplicate")):
                payload = learning.normalize_escalation(_escalation())
                upserted = dict(payload)
                upserted["status"] = "upserted"
                learning.write_to_ledger(upserted)

                learning._process_learning_sync(_escalation())

            lines = ledger.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"status": "duplicate"', lines[-1])

    def test_upsert_to_pinecone_uses_approved_namespace(self):
        calls = {}

        class FakeIndex:
            def upsert(self, vectors, namespace=None):
                calls["vectors"] = vectors
                calls["namespace"] = namespace

        class FakePinecone:
            def __init__(self, api_key):
                calls["api_key"] = api_key

            def Index(self, index_name):
                calls["index_name"] = index_name
                return FakeIndex()

        payload = learning.normalize_escalation(_escalation())
        with patch.dict("os.environ", {"PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}), \
             patch.object(learning, "Pinecone", FakePinecone):
            learning.upsert_to_pinecone(payload, [0.1, 0.2])

        self.assertEqual(calls["namespace"], "approved_escalation_qa")
        self.assertEqual(calls["index_name"], "test-index")
        self.assertEqual(calls["vectors"][0]["id"], payload["id"])
        self.assertEqual(calls["vectors"][0]["metadata"]["approval_state"], "owner_approved")


if __name__ == "__main__":
    unittest.main()
