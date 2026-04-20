"""Approved escalation Q&A learning pipeline.

Best-effort path: owner-approved Telegram replies are normalized, written to an
append-only ledger, embedded, and upserted into a separate Pinecone namespace.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

logger = logging.getLogger(__name__)

APPROVED_QA_NAMESPACE = "approved_escalation_qa"
SCHEMA_VERSION = "approved_escalation_qa.v1"
EMBED_MODEL = "text-embedding-3-small"
BACKEND_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = BACKEND_ROOT / "ai" / "data" / "approved_escalation_qa.jsonl"

_SHORT_REPLIES = {"yes", "no", "yep", "yeah", "nope", "ok", "okay", "sure"}
_AMBIGUOUS_PHRASES = {
    "not sure",
    "i don't know",
    "i dont know",
    "ask them",
    "call me",
    "call us",
    "send pictures",
    "send pics",
}


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat() + "Z"
    return str(value)


def _derive_topic(question: str) -> str:
    q = question.lower()
    if any(term in q for term in ("will it work", "will this work", "fit my", "fit on", "compatible", "hitch", "vehicle", "cr-v", "crv", "honda", "toyota", "ford", "bike", "e-bike", "ebike")):
        return "fitment"
    if any(term in q for term in ("warranty", "liability", "covered", "cover", "void")):
        return "warranty"
    if any(term in q for term in ("safe", "safety", "damage", "dangerous", "risk", "fall off", "battery", "legal", "law")):
        return "safety_damage"
    if any(term in q for term in ("color", "colour", "dimension", "weight", "height", "width", "length", "spec", "come in")):
        return "product_info"
    return "general"


def _extract_fitment(question: str) -> dict[str, Any]:
    q = question.lower()
    metadata: dict[str, Any] = {
        "product": None,
        "vehicle": None,
        "vehicle_year": None,
        "bike_type": None,
        "bike_count": None,
        "hitch": None,
    }

    if "honda" in q and ("cr-v" in q or "crv" in q):
        metadata["vehicle"] = "Honda CR-V"
    elif "cr-v" in q or "crv" in q:
        metadata["vehicle"] = "Honda CR-V"

    year_match = re.search(r"\b(19|20)\d{2}\b", q)
    if year_match:
        metadata["vehicle_year"] = year_match.group(0)

    if any(term in q for term in ("standard ebike", "standard e-bike", "ebike", "e-bike", "electric bike")):
        metadata["bike_type"] = "ebike"
    elif "bike" in q:
        metadata["bike_type"] = "bike"

    if re.search(r"\b(?:one|1|single)\b", q):
        metadata["bike_count"] = 1
    elif re.search(r"\b(?:two|2)\b", q):
        metadata["bike_count"] = 2

    if any(term in q for term in ("2-inch hitch", "2 inch hitch", '2" hitch', "2-inch receiver", "2 inch receiver")):
        metadata["hitch"] = "2-inch"

    return metadata


def should_learn(owner_reply: str) -> bool:
    """Return True when an owner reply is suitable for approved Q&A learning."""
    reply = _clean_text(owner_reply)
    lower = reply.lower()

    if not reply:
        return False
    if lower.startswith("nolearn:") or lower.startswith("[no learn]"):
        return False
    if lower in _SHORT_REPLIES:
        return False
    if len(reply) < 24:
        return False
    if any(phrase in lower for phrase in _AMBIGUOUS_PHRASES):
        return False
    if re.fullmatch(r"https?://\S+", reply):
        return False
    return True


def normalize_escalation(escalation) -> dict:
    """Normalize an escalation object into the approved Q&A schema."""
    question = _clean_text(getattr(escalation, "user_question", ""))
    answer = _clean_text(getattr(escalation, "owner_reply", ""))
    topic = _derive_topic(question)
    fitment = _extract_fitment(question)

    payload = {
        "id": None,
        "question_text": question,
        "answer_text": answer,
        "embedding_text": "",
        "topic": topic,
        **fitment,
        "source": "telegram_escalation",
        "approval_state": "owner_approved",
        "escalation_id": getattr(escalation, "escalation_id", None),
        "conversation_id": getattr(escalation, "conversation_id", None),
        "source_url": getattr(escalation, "source_url", None),
        "created_at": _iso(getattr(escalation, "created_at", None)),
        "approved_at": _iso(getattr(escalation, "owner_replied_at", None)),
        "learned_at": _utc_now_iso(),
        "owner_reply_message_id": None,
        "telegram_chat_id": getattr(escalation, "telegram_chat_id", None),
        "content_hash": None,
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
    }
    payload["embedding_text"] = _build_embedding_text(payload)
    payload["content_hash"] = generate_content_hash(payload)
    payload["id"] = f"approved_qa_{payload['content_hash']}"
    return payload


def _build_embedding_text(payload: dict) -> str:
    fitment_parts = []
    for key in ("vehicle_year", "vehicle", "bike_count", "bike_type", "hitch"):
        value = payload.get(key)
        if value not in (None, ""):
            fitment_parts.append(f"{key}: {value}")
    fitment_text = "; ".join(fitment_parts) if fitment_parts else "None"
    return (
        "Approved owner Q&A\n"
        f"Topic: {payload.get('topic') or 'general'}\n"
        f"Question: {payload.get('question_text') or ''}\n"
        f"Answer: {payload.get('answer_text') or ''}\n"
        f"Fitment: {fitment_text}"
    )


def generate_content_hash(payload: dict) -> str:
    """Generate stable hash from normalized question, answer, and topic."""
    canonical = json.dumps(
        {
            "question_text": _clean_text(payload.get("question_text", "")).lower(),
            "answer_text": _clean_text(payload.get("answer_text", "")).lower(),
            "topic": _clean_text(payload.get("topic", "general")).lower(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _ledger_has_hash(content_hash: str) -> bool:
    if not LEDGER_PATH.exists():
        return False
    with LEDGER_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if existing.get("content_hash") == content_hash and existing.get("status") == "upserted":
                return True
    return False


def write_to_ledger(payload: dict) -> None:
    """Append the full normalized payload to the JSONL ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def embed_payload(payload: dict) -> list[float]:
    """Embed the approved Q&A using the same OpenAI embedding model as RAG."""
    client = OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=payload["embedding_text"])
    return response.data[0].embedding


def upsert_to_pinecone(payload: dict, vector: list[float]) -> None:
    """Upsert the approved Q&A vector into the separate approved namespace."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "radgnarack-assist")
    if not api_key or api_key == "your-pinecone-api-key-here":
        raise RuntimeError("Pinecone not initialized. Check PINECONE_API_KEY.")

    metadata = {
        "text": payload["embedding_text"],
        "question_text": payload["question_text"],
        "answer_text": payload["answer_text"],
        "topic": payload["topic"],
        "product": payload.get("product"),
        "vehicle": payload.get("vehicle"),
        "vehicle_year": payload.get("vehicle_year"),
        "bike_type": payload.get("bike_type"),
        "bike_count": payload.get("bike_count"),
        "hitch": payload.get("hitch"),
        "source": payload["source"],
        "approval_state": payload["approval_state"],
        "escalation_id": payload.get("escalation_id"),
        "conversation_id": payload.get("conversation_id"),
        "source_url": payload.get("source_url"),
        "created_at": payload.get("created_at"),
        "approved_at": payload.get("approved_at"),
        "learned_at": payload.get("learned_at"),
        "content_hash": payload["content_hash"],
        "schema_version": payload["schema_version"],
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    index.upsert(
        vectors=[{"id": payload["id"], "values": vector, "metadata": metadata}],
        namespace=APPROVED_QA_NAMESPACE,
    )


def _process_learning_sync(escalation) -> None:
    if not should_learn(getattr(escalation, "owner_reply", "")):
        return

    payload = normalize_escalation(escalation)
    if _ledger_has_hash(payload["content_hash"]):
        duplicate_payload = dict(payload)
        duplicate_payload["status"] = "duplicate"
        write_to_ledger(duplicate_payload)
        return

    write_to_ledger(payload)

    try:
        vector = embed_payload(payload)
        upsert_to_pinecone(payload, vector)
    except Exception:
        failed_payload = dict(payload)
        failed_payload["status"] = "failed"
        failed_payload["failed_at"] = _utc_now_iso()
        write_to_ledger(failed_payload)
        raise

    upserted_payload = dict(payload)
    upserted_payload["status"] = "upserted"
    upserted_payload["upserted_at"] = _utc_now_iso()
    write_to_ledger(upserted_payload)


def process_learning(escalation) -> None:
    """Run best-effort learning synchronously inside the caller's try/except."""
    _process_learning_sync(escalation)
