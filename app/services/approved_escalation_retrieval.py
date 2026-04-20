"""Strict retrieval for owner-approved escalation Q&A."""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

logger = logging.getLogger(__name__)

APPROVED_QA_NAMESPACE = "approved_escalation_qa"
EMBED_MODEL = "text-embedding-3-small"
APPROVED_QA_THRESHOLD = 0.50
APPROVED_QA_MARGIN = 0.04


def _clean(value) -> str:
    return str(value or "").strip().lower()


def _metadata_matches(metadata: dict, context: dict) -> bool:
    topic = _clean(context.get("topic"))
    match_topic = _clean(metadata.get("topic"))
    if topic and topic != "general" and match_topic and match_topic != topic:
        return False

    fitment = context.get("fitment") or {}
    for key in ("vehicle", "vehicle_year", "bike_type", "hitch"):
        expected = _clean(fitment.get(key) or context.get(key))
        actual = _clean(metadata.get(key))
        if expected and actual and expected != actual:
            return False

    expected_count = fitment.get("bike_count") or context.get("bike_count")
    actual_count = metadata.get("bike_count")
    if expected_count not in (None, "") and actual_count not in (None, ""):
        if str(expected_count) != str(actual_count):
            return False

    return True


def get_approved_answer(query: str, context: dict) -> Optional[dict]:
    """Return the single accepted approved Q&A match, or None."""
    logger.warning("RETRIEVAL START query=%r", query)
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "radgnarack-assist")
    if not api_key or api_key == "your-pinecone-api-key-here":
        return None

    try:
        embedding = OpenAI().embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
        index = Pinecone(api_key=api_key).Index(index_name)
        results = index.query(
            vector=embedding,
            top_k=2,
            include_metadata=True,
            namespace=APPROVED_QA_NAMESPACE,
        )
    except Exception:
        logger.error("Approved escalation retrieval failed", exc_info=True)
        return None

    if isinstance(results, dict):
        matches = list(results.get("matches", []))
    else:
        matches = list(getattr(results, "matches", []) or [])
    logger.warning("RETRIEVAL RESULTS count=%d", len(matches))
    if not matches:
        logger.warning("RETRIEVAL REJECT reason=no_matches")
        return None

    def _score(match) -> float:
        if isinstance(match, dict):
            return float(match.get("score", 0.0) or 0.0)
        return float(getattr(match, "score", 0.0) or 0.0)

    def _metadata(match) -> dict:
        if isinstance(match, dict):
            return dict(match.get("metadata", {}) or {})
        return dict(getattr(match, "metadata", {}) or {})

    top = matches[0]
    top_score = _score(top)
    metadata = _metadata(top)
    logger.warning("RETRIEVAL TOP score=%s metadata=%s", top_score, metadata.get("content_hash"))
    if top_score < APPROVED_QA_THRESHOLD:
        logger.warning("RETRIEVAL REJECT reason=threshold score=%s threshold=%s", top_score, APPROVED_QA_THRESHOLD)
        return None

    if len(matches) > 1:
        second_score = _score(matches[1])
        margin = top_score - second_score
        logger.warning(
            "RETRIEVAL MARGIN top_score=%s second_score=%s margin=%s required_margin=%s",
            top_score,
            second_score,
            margin,
            APPROVED_QA_MARGIN,
        )

    if metadata.get("approval_state") != "owner_approved":
        logger.warning("RETRIEVAL REJECT reason=approval_state value=%r", metadata.get("approval_state"))
        return None
    if not metadata.get("answer_text"):
        logger.warning("RETRIEVAL REJECT reason=missing_answer_text")
        return None
    if not _metadata_matches(metadata, context or {}):
        logger.warning("RETRIEVAL REJECT reason=metadata top_metadata=%r context=%r", metadata, context or {})
        return None

    selected_score = top_score
    selected_metadata = metadata
    valid_candidates = [(top_score, metadata)]
    for match in matches[1:]:
        candidate_score = _score(match)
        candidate_metadata = _metadata(match)
        if candidate_score < APPROVED_QA_THRESHOLD:
            continue
        if candidate_metadata.get("approval_state") != "owner_approved":
            continue
        if not candidate_metadata.get("answer_text"):
            continue
        if not _metadata_matches(candidate_metadata, context or {}):
            continue
        valid_candidates.append((candidate_score, candidate_metadata))

    if _clean((context or {}).get("topic")) == "product_info" and len(valid_candidates) > 1:
        close_candidates = [
            candidate
            for candidate in valid_candidates
            if top_score - candidate[0] <= 0.02
        ]
        selected_score, selected_metadata = max(
            close_candidates,
            key=lambda candidate: len(str(candidate[1].get("answer_text") or "")),
        )
        if selected_metadata is not metadata:
            logger.warning(
                "RETRIEVAL TIEBREAK selected_longer_answer id=%s score=%s answer_length=%s",
                selected_metadata.get("content_hash"),
                selected_score,
                len(str(selected_metadata.get("answer_text") or "")),
            )

    logger.warning("RETRIEVAL ACCEPT id=%s score=%s", selected_metadata.get("content_hash"), selected_score)
    return {
        "answer_text": selected_metadata["answer_text"],
        "question_text": selected_metadata.get("question_text"),
        "score": selected_score,
        "metadata": selected_metadata,
    }
