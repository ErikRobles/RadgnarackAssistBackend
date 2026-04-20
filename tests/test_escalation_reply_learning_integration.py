import unittest
from unittest.mock import patch

from app.repositories.escalation_repository import EscalationRepository
from app.schemas.escalation import EscalationStatus
from app.services import escalation_service


class EscalationReplyLearningIntegrationTests(unittest.TestCase):
    def test_process_owner_reply_still_records_reply_when_learning_fails(self):
        repo = EscalationRepository()
        escalation = repo.create(
            user_question="What color does the rack come in?",
            conversation_id="conv-1",
        )
        repo.update_status(escalation.escalation_id, EscalationStatus.SENT_TO_OWNER)

        def fail_learning(updated):
            raise RuntimeError("learning failure")

        with patch.object(escalation_service, "escalation_repo", repo), \
             patch.object(escalation_service, "process_learning", fail_learning):
            updated = escalation_service.process_owner_reply(
                escalation.escalation_id,
                "The rack comes in black powder coat finish.",
            )

        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, EscalationStatus.OWNER_REPLIED)
        self.assertEqual(updated.owner_reply, "The rack comes in black powder coat finish.")


if __name__ == "__main__":
    unittest.main()
