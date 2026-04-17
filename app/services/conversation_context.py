"""
Minimal in-memory conversation context for fitment clarification follow-ups.
Maps conversation_id to last assistant response status.
"""
from dataclasses import dataclass, field
from typing import Optional
import time
import re


@dataclass
class ConversationState:
    """Tracks the state of a conversation."""
    conversation_id: str
    last_question: str
    last_status: str
    last_answer: str
    last_intent: str | None = None
    last_turn_type: str | None = None
    clarification_attempts: int = 0
    fitment_context: str = ""
    timestamp: float = field(default_factory=time.time)


# In-memory storage - conversation_id -> ConversationState
# In production, this should be Redis or similar
_conversation_states: dict[str, ConversationState] = {}

# Time-to-live for conversation state (5 minutes)
STATE_TTL_SECONDS = 300


def _cleanup_expired():
    """Remove expired conversation states."""
    now = time.time()
    expired = [
        cid for cid, state in _conversation_states.items()
        if now - state.timestamp > STATE_TTL_SECONDS
    ]
    for cid in expired:
        del _conversation_states[cid]


def get_conversation_state(conversation_id: str) -> Optional[ConversationState]:
    """Get the state for a conversation if it exists and is not expired."""
    _cleanup_expired()
    return _conversation_states.get(conversation_id)


def set_conversation_state(
    conversation_id: str,
    question: str,
    status: str,
    answer: str,
    intent: str | None = None,
    turn_type: str | None = None,
    clarification_attempts: int = 0,
    fitment_context: str = "",
) -> None:
    """Store the state for a conversation."""
    resolved_turn_type = turn_type
    if resolved_turn_type is None and _looks_like_clarification_prompt(answer, status):
        resolved_turn_type = "clarification"

    _conversation_states[conversation_id] = ConversationState(
        conversation_id=conversation_id,
        last_question=question,
        last_status=status,
        last_answer=answer,
        last_intent=intent,
        last_turn_type=resolved_turn_type,
        clarification_attempts=clarification_attempts,
        fitment_context=fitment_context,
    )


def _looks_like_clarification_prompt(answer: str, status: str | None = None) -> bool:
    """Detect prior assistant clarification without depending on exact prompt text."""
    if status == "clarification_needed":
        return True

    answer_l = (answer or "").lower()
    clarification_keywords = ("year", "bike", "vehicle", "model", "type")
    return "?" in answer_l and any(keyword in answer_l for keyword in clarification_keywords)


def _looks_like_structured_follow_up(message: str) -> bool:
    """
    Leniently detect short clarification replies containing vehicle/bike details.
    Examples: "2020 CRV and one standard ebike", "Honda CR-V 2020", "standard e-bike".
    """
    q = (message or "").lower()
    words = re.findall(r"[a-z0-9-]+", q)
    if not words or len(words) >= 20:
        return False

    has_number = bool(re.search(r"\b(?:19|20)\d{2}\b|\b\d+\b", q))
    vehicle_terms = (
        "crv", "cr-v", "honda", "toyota", "ford", "chevy", "subaru", "bmw",
        "audi", "mercedes", "rav4", "outback", "civic", "camry", "f150",
        "f-150", "car", "suv", "truck", "van", "vehicle",
    )
    bike_terms = (
        "ebike", "e-bike", "electric", "bike", "bicycle", "standard",
        "fat", "tire", "step-through", "stepthrough",
    )
    has_vehicle = any(term in q for term in vehicle_terms)
    has_bike = any(term in q for term in bike_terms)

    return has_number or has_vehicle or has_bike


def is_follow_up_to_clarification(
    conversation_id: Optional[str],
    message: Optional[str] = None,
) -> bool:
    """
    Check if this conversation had a prior clarification response and the new
    message looks like a short structured clarification reply.
    """
    if not conversation_id:
        return False
    state = get_conversation_state(conversation_id)
    if not state:
        return False

    previous_turn_is_clarification = (
        state.last_turn_type == "clarification"
        or _looks_like_clarification_prompt(state.last_answer, state.last_status)
    )
    if not previous_turn_is_clarification:
        return False

    if message is None:
        return True

    msg = (message or "").lower().strip()
    yes_words = {
        "yes", "yes i do", "yes, i do", "yep", "yeah", "correct",
        "it does", "yes it does", "yes, it does",
    }
    if any(yes_word == msg for yes_word in yes_words) and "hitch" in (state.last_answer or "").lower():
        return True

    return _looks_like_structured_follow_up(message)


def build_enriched_fitment_query(
    conversation_id: str,
    follow_up_question: str
) -> Optional[str]:
    """
    Build an enriched fitment query combining original + follow-up.
    Returns None if conversation state not found.
    """
    state = get_conversation_state(conversation_id)
    if not state:
        return None
    
    original = (state.fitment_context or state.last_question).strip().rstrip(".?!")
    follow_up = follow_up_question.strip().rstrip(".?!")
    
    intent_label = state.last_intent or "fitment compatibility"
    enriched = (
        f"{intent_label} question. "
        f"Original question: {original}. "
        f"User clarification: {follow_up}."
    )
    
    return enriched


def clear_conversation_state(conversation_id: str) -> None:
    """Clear state for a conversation (e.g., after escalation or successful answer)."""
    if conversation_id in _conversation_states:
        del _conversation_states[conversation_id]
