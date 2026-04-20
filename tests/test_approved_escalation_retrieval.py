import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services import approved_escalation_retrieval as retrieval


class FakeEmbeddingResponse:
    data = [SimpleNamespace(embedding=[0.1, 0.2, 0.3])]


class FakeOpenAI:
    class embeddings:
        @staticmethod
        def create(model, input):
            return FakeEmbeddingResponse()


def _match(score, **metadata):
    base = {
        "approval_state": "owner_approved",
        "answer_text": "The rack comes in black powder coat finish.",
        "question_text": "What color does the rack come in?",
        "topic": "product_info",
    }
    base.update(metadata)
    return {"score": score, "metadata": base}


class ApprovedEscalationRetrievalTests(unittest.TestCase):
    def _patch_query(self, matches):
        calls = {}

        class FakeIndex:
            def query(self, **kwargs):
                calls.update(kwargs)
                return {"matches": matches}

        class FakePinecone:
            def __init__(self, api_key):
                pass

            def Index(self, index_name):
                calls["index_name"] = index_name
                return FakeIndex()

        return calls, patch.dict("os.environ", {"PINECONE_API_KEY": "test-key", "PINECONE_INDEX_NAME": "test-index"}), patch.object(retrieval, "OpenAI", FakeOpenAI), patch.object(retrieval, "Pinecone", FakePinecone)

    def test_get_approved_answer_returns_top_match_when_strictly_valid(self):
        calls, env_patch, openai_patch, pinecone_patch = self._patch_query([_match(0.90), _match(0.80)])
        with env_patch, openai_patch, pinecone_patch:
            result = retrieval.get_approved_answer("What color is it?", {"topic": "product_info"})

        self.assertEqual(result["answer_text"], "The rack comes in black powder coat finish.")
        self.assertEqual(result["score"], 0.90)
        self.assertEqual(calls["namespace"], "approved_escalation_qa")
        self.assertEqual(calls["top_k"], 2)

    def test_get_approved_answer_rejects_low_score(self):
        calls, env_patch, openai_patch, pinecone_patch = self._patch_query([_match(0.49), _match(0.40)])
        with env_patch, openai_patch, pinecone_patch:
            self.assertIsNone(retrieval.get_approved_answer("What color is it?", {"topic": "product_info"}))

    def test_get_approved_answer_accepts_close_margin_duplicates(self):
        calls, env_patch, openai_patch, pinecone_patch = self._patch_query([_match(0.90), _match(0.87)])
        with env_patch, openai_patch, pinecone_patch:
            result = retrieval.get_approved_answer("What color is it?", {"topic": "product_info"})

        self.assertEqual(result["answer_text"], "The rack comes in black powder coat finish.")
        self.assertEqual(result["score"], 0.90)

    def test_get_approved_answer_rejects_fitment_metadata_mismatch(self):
        calls, env_patch, openai_patch, pinecone_patch = self._patch_query([_match(0.91, topic="fitment", vehicle="Honda CR-V"), _match(0.70)])
        with env_patch, openai_patch, pinecone_patch:
            result = retrieval.get_approved_answer(
                "Will this fit my Toyota?",
                {"topic": "fitment", "fitment": {"vehicle": "Toyota RAV4"}},
            )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
