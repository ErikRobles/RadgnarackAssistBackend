import json
import os
import random
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key-here"

from app.api.v1 import chat as chat_module
from app.api.v1 import escalation as escalation_api
from app.repositories.escalation_repository import EscalationRepository
from app.schemas.chat import ChatRequest
from app.schemas.escalation import EscalationStatus
from app.services import approved_escalation_learning as learning
from app.services import conversation_context
from app.services import escalation_service


def _fake_result(question: str, *, status="answered", escalation_needed=False, answer=None):
    answer_text = answer if answer is not None else f"Answer for {question}"
    result = SimpleNamespace(status=status)
    result_dict = {
        "question": question,
        "answer": answer_text,
        "sources": [],
        "used_context": status == "answered",
        "escalation_needed": escalation_needed,
        "status": status,
        "retrieved_chunks": [],
    }
    return result, result_dict


class ConcurrencyProductionSafetyTests(unittest.TestCase):
    def setUp(self):
        conversation_context._conversation_states.clear()

    def test_a_conversation_isolation_with_20_concurrent_chat_requests(self):
        def fake_answer_question(question):
            return SimpleNamespace(status="answered", question=question)

        def fake_result_to_dict(result):
            return _fake_result(result.question, answer=f"isolated response for {result.question}")[1]

        def send(i: int):
            conv_id = f"conv_iso_{i}"
            request = ChatRequest(question=f"Question {i}", conversation_id=conv_id)
            response = chat_module.chat(request)
            state = conversation_context.get_conversation_state(conv_id)
            return conv_id, response, state

        with patch.object(chat_module, "get_approved_answer", return_value=None), \
             patch.object(chat_module, "answer_question", side_effect=fake_answer_question), \
             patch.object(chat_module, "result_to_dict", side_effect=fake_result_to_dict):
            with ThreadPoolExecutor(max_workers=20) as executor:
                results = [future.result() for future in as_completed([executor.submit(send, i) for i in range(20)])]

        self.assertEqual(len(results), 20)
        self.assertEqual(len({conv_id for conv_id, _, _ in results}), 20)
        for conv_id, response, state in results:
            self.assertIsNotNone(state)
            self.assertEqual(state.conversation_id, conv_id)
            self.assertIn(state.last_question, response.answer)
            self.assertTrue(response.answer.startswith("isolated response for Question"))

    def test_b_escalation_creation_race_unique_ids_and_conversation_mapping(self):
        repo = EscalationRepository()

        def create(i: int):
            return repo.create(user_question=f"Escalation question {i}", conversation_id=f"conv_esc_{i}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            escalations = [future.result() for future in as_completed([executor.submit(create, i) for i in range(10)])]

        self.assertEqual(len(escalations), 10)
        self.assertEqual(len({esc.escalation_id for esc in escalations}), 10)
        for esc in escalations:
            self.assertEqual(repo.get(esc.escalation_id).conversation_id, esc.conversation_id)

    def test_c_telegram_reply_routing_with_shuffled_owner_replies(self):
        repo = EscalationRepository()
        escalations = []
        for i in range(10):
            esc = repo.create(user_question=f"Question {i}", conversation_id=f"conv_reply_{i}")
            repo.update_status(esc.escalation_id, EscalationStatus.SENT_TO_OWNER)
            repo.update_telegram_message_id(esc.escalation_id, telegram_message_id=1000 + i, telegram_chat_id="owner-chat")
            escalations.append(esc)

        reply_order = list(range(10))
        random.Random(123).shuffle(reply_order)

        def reply(i: int):
            esc = repo.get_by_telegram_message_id("owner-chat", 1000 + i)
            return escalation_service.process_owner_reply(esc.escalation_id, f"Owner answer {i}")

        with patch.object(escalation_service, "escalation_repo", repo), \
             patch.object(escalation_service, "process_learning", return_value=None):
            with ThreadPoolExecutor(max_workers=10) as executor:
                updated = [future.result() for future in as_completed([executor.submit(reply, i) for i in reply_order])]

        self.assertEqual(len(updated), 10)
        for i, esc in enumerate(escalations):
            stored = repo.get(esc.escalation_id)
            self.assertEqual(stored.owner_reply, f"Owner answer {i}")
            self.assertEqual(stored.conversation_id, f"conv_reply_{i}")
            self.assertEqual(stored.status, EscalationStatus.OWNER_REPLIED)

    def test_d_polling_isolation_returns_only_matching_conversation_reply(self):
        repo = EscalationRepository()
        for i in range(10):
            esc = repo.create(user_question=f"Poll question {i}", conversation_id=f"conv_poll_{i}")
            repo.add_owner_reply(esc.escalation_id, f"Poll answer {i}")

        def poll(i: int):
            return escalation_api.check_escalation_status(f"conv_poll_{i}")

        with patch.object(escalation_api, "escalation_repo", repo):
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = {i: future.result() for i, future in [(i, executor.submit(poll, i)) for i in range(10)]}

        for i, response in results.items():
            self.assertTrue(response["has_reply"])
            self.assertEqual(response["owner_reply"], f"Poll answer {i}")
            self.assertEqual(response["user_question"], f"Poll question {i}")

    def test_e_approved_learning_concurrency_jsonl_valid_and_correct_ids(self):
        def make_escalation(i: int):
            return SimpleNamespace(
                escalation_id=f"esc_learn_{i:05d}",
                conversation_id=f"conv_learn_{i}",
                user_question=f"What is approved learning question {i}?",
                source_url="https://example.com",
                created_at=datetime(2026, 4, 20, 12, 0, 0),
                owner_reply=f"This is a complete owner-approved learning answer number {i}.",
                owner_replied_at=datetime(2026, 4, 20, 12, 1, 0),
                telegram_chat_id="owner-chat",
            )

        with TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "approved_escalation_qa.jsonl"
            with patch.object(learning, "LEDGER_PATH", ledger), \
                 patch.object(learning, "embed_payload", return_value=[0.1, 0.2, 0.3]), \
                 patch.object(learning, "upsert_to_pinecone", return_value=None):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    for future in as_completed([executor.submit(learning._process_learning_sync, make_escalation(i)) for i in range(10)]):
                        future.result()

            lines = ledger.read_text().splitlines()
            self.assertEqual(len(lines), 20)  # pending + upserted for each learned reply
            records = [json.loads(line) for line in lines]
            self.assertTrue(all(record["escalation_id"].startswith("esc_learn_") for record in records))
            upserted_ids = {record["escalation_id"] for record in records if record["status"] == "upserted"}
            self.assertEqual(upserted_ids, {f"esc_learn_{i:05d}" for i in range(10)})

    def test_f_approved_retrieval_isolation_approved_hit_and_escalation_flow(self):
        created_escalations = []

        def fake_get_approved_answer(query, context):
            if "approved" in query.lower():
                return {
                    "answer_text": "Approved answer for user A only.",
                    "score": 0.91,
                    "metadata": {"content_hash": "hash-a"},
                }
            return None

        def fake_answer_question(question):
            if "escalate" in question.lower():
                return SimpleNamespace(status="insufficient_context")
            return SimpleNamespace(status="answered")

        def fake_result_to_dict(result):
            if result.status == "insufficient_context":
                return _fake_result(
                    "escalate",
                    status="insufficient_context",
                    escalation_needed=True,
                    answer="I'm not sure based on the available information.",
                )[1]
            return _fake_result("rag", answer="RAG answer")[1]

        def fake_create_escalation(**kwargs):
            created_escalations.append(kwargs)
            return SimpleNamespace(escalation_id="esc_created")

        requests = [
            ChatRequest(question="Approved question for user A", conversation_id="conv_user_a"),
            ChatRequest(question="Escalate question for user B", conversation_id="conv_user_b"),
        ]

        with patch.object(chat_module, "get_approved_answer", side_effect=fake_get_approved_answer), \
             patch.object(chat_module, "answer_question", side_effect=fake_answer_question), \
             patch.object(chat_module, "result_to_dict", side_effect=fake_result_to_dict), \
             patch.object(chat_module, "create_escalation", side_effect=fake_create_escalation):
            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = [future.result() for future in as_completed([executor.submit(chat_module.chat, request) for request in requests])]

        answers = {response.answer for response in responses}
        self.assertIn("Approved answer for user A only.", answers)
        self.assertIn("I'm not sure based on the available information.", answers)
        self.assertEqual(len(created_escalations), 1)
        self.assertEqual(created_escalations[0]["conversation_id"], "conv_user_b")

    def test_step_3_load_simulation_20_concurrent_chat_requests_mixed_paths(self):
        created_escalations = []
        created_lock = threading.Lock()

        def fake_get_approved_answer(query, context):
            if query.startswith("approved"):
                return {
                    "answer_text": f"Approved answer for {query}",
                    "score": 0.9,
                    "metadata": {"content_hash": query},
                }
            return None

        def fake_answer_question(question):
            if question.startswith("escalate"):
                return SimpleNamespace(status="insufficient_context")
            return SimpleNamespace(status="answered")

        def fake_result_to_dict(result):
            if result.status == "insufficient_context":
                return _fake_result(
                    "escalate",
                    status="insufficient_context",
                    escalation_needed=True,
                    answer="I'm not sure based on the available information.",
                )[1]
            return _fake_result("rag", answer="RAG path answer")[1]

        def fake_create_escalation(**kwargs):
            with created_lock:
                created_escalations.append(kwargs)
            return SimpleNamespace(escalation_id=f"esc_load_{len(created_escalations)}")

        questions = []
        for i in range(20):
            if i % 3 == 0:
                questions.append(f"approved load question {i}")
            elif i % 3 == 1:
                questions.append(f"rag load question {i}")
            else:
                questions.append(f"escalate load question {i}")

        def send(i: int):
            return chat_module.chat(ChatRequest(question=questions[i], conversation_id=f"conv_load_{i}"))

        with patch.object(chat_module, "get_approved_answer", side_effect=fake_get_approved_answer), \
             patch.object(chat_module, "answer_question", side_effect=fake_answer_question), \
             patch.object(chat_module, "result_to_dict", side_effect=fake_result_to_dict), \
             patch.object(chat_module, "create_escalation", side_effect=fake_create_escalation):
            with ThreadPoolExecutor(max_workers=20) as executor:
                responses = [future.result() for future in as_completed([executor.submit(send, i) for i in range(20)])]

        self.assertEqual(len(responses), 20)
        self.assertEqual(len(created_escalations), len([q for q in questions if q.startswith("escalate")]))
        self.assertEqual(len({item["conversation_id"] for item in created_escalations}), len(created_escalations))
        self.assertTrue(any(response.answer.startswith("Approved answer") for response in responses))
        self.assertTrue(any(response.answer == "RAG path answer" for response in responses))
        self.assertTrue(any(response.escalation_needed for response in responses))


if __name__ == "__main__":
    unittest.main()
