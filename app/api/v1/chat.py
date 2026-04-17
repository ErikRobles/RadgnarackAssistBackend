from fastapi import APIRouter, HTTPException
import os
import logging
import re

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.escalation_service import create_escalation, should_escalate
from app.repositories.escalation_repository import escalation_repo
from app.services.conversation_context import (
    is_follow_up_to_clarification,
    build_enriched_fitment_query,
    set_conversation_state,
    get_conversation_state,
    clear_conversation_state,
)

# Use Pinecone if API key is configured, otherwise fall back to local JSON
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY and PINECONE_API_KEY != "your-pinecone-api-key-here":
    from ai.services.pinecone_rag_service import answer_question, result_to_dict
else:
    from ai.services.rag_service import answer_question, result_to_dict

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _looks_safety_critical(question: str) -> bool:
    """Keep safety above follow-up enrichment so safety still reaches normal escalation."""
    q = question.lower()
    safety_keywords = [
        "battery", "remove battery", "detach battery",
        "safety", "safe", "dangerous", "risk", "damage",
        "warranty", "void warranty", "insurance",
        "legal", "law", "requirement", "required by law",
        "secure", "loose", "fall off", "detach", "come off",
    ]
    return any(keyword in q for keyword in safety_keywords)


def _looks_fitment_question(question: str) -> bool:
    """Detect fitment/compatibility questions without mutating the input."""
    q = question.lower()
    intent_terms = (
        "will it work", "will this work", "work with", "work for",
        "fit my", "fit on", "compatible with", "does this fit",
        "will this fit", "will it fit", "can i use",
    )
    target_terms = (
        "crv", "cr-v", "honda", "toyota", "ford", "chevy", "subaru",
        "suv", "truck", "vehicle", "car", "ebike", "e-bike", "bike",
    )
    return any(term in q for term in intent_terms) and any(term in q for term in target_terms)


def _build_initial_fitment_clarification(question: str) -> str:
    q = question.lower()
    if any(term in q for term in ("crv", "cr-v", "honda", "toyota", "ford", "chevy", "subaru")):
        return "I can help with that. What year is your vehicle? And what type of bike or e-bike are you planning to carry?"
    return "I can help with that. Which vehicle, hitch size, and bike type are you using?"


def _build_fitment_fallback_clarification(question: str) -> str:
    """Provide conditional fitment guidance without claiming exact compatibility."""
    q = question.lower()
    vehicle = "your vehicle"
    if "cr-v" in q or "crv" in q:
        vehicle = "your 2020 Honda CR-V" if "2020" in q else "your Honda CR-V"

    bike = "your bike"
    if "ebike" in q or "e-bike" in q or "electric" in q:
        bike = "your standard e-bike" if "standard" in q else "your e-bike"

    return (
        f"Based on {vehicle} and {bike}, compatibility depends on your hitch type, "
        "rack model, and weight limits. Do you know if your vehicle has a 2-inch hitch installed?"
    )


def _is_affirmative_hitch_reply(question: str, previous_answer: str) -> bool:
    q = question.lower().strip()
    affirmative_replies = {
        "yes", "yes i do", "yes, i do", "yeah", "yep",
        "yes it does", "yes, it does", "it does",
    }
    previous_asked_hitch = "hitch" in previous_answer.lower() and (
        "2-inch" in previous_answer.lower() or "2 inch" in previous_answer.lower()
    )
    return previous_asked_hitch and q in affirmative_replies


def _normalize_fitment_context(
    text: str,
    current_question: str = "",
    previous_answer: str = "",
) -> dict[str, object]:
    """Normalize accumulated fitment context into structured fields."""
    q = text.lower()
    normalized: dict[str, object] = {}

    if "honda" in q and ("crv" in q or "cr-v" in q):
        normalized["vehicle"] = "Honda CR-V"
    elif "crv" in q or "cr-v" in q:
        normalized["vehicle"] = "Honda CR-V"

    year_match = re.search(r"\b(19|20)\d{2}\b", q)
    if year_match:
        normalized["vehicle_year"] = year_match.group(0)

    if any(term in q for term in ("standard ebike", "standard e-bike", "ebike", "e-bike", "electric bike")):
        normalized["bike_type"] = "ebike"

    if re.search(r"\b(?:one|1|single)\b", q):
        normalized["bike_count"] = 1
    elif "bike_type" in normalized and not re.search(r"\b(?:two|three|four)\b|\b[2-9]\d*\s*(?:bike|bikes|ebike|ebikes|e-bike|e-bikes)\b", q):
        normalized["bike_count"] = 1

    if (
        "2-inch hitch" in q
        or "2 inch hitch" in q
        or "2\" hitch" in q
        or "2-inch receiver" in q
        or "2 inch receiver" in q
        or _is_affirmative_hitch_reply(current_question, previous_answer)
    ):
        normalized["hitch"] = "2-inch"

    return normalized


def _has_complete_fitment_data(fitment_context: dict[str, object]) -> bool:
    """Detect enough structured fitment data for a safe recommendation."""
    return (
        bool(fitment_context.get("vehicle"))
        and int(fitment_context.get("bike_count") or 0) >= 1
        and bool(fitment_context.get("bike_type"))
        and fitment_context.get("hitch") == "2-inch"
    )


def _get_missing_fitment_fields(fitment_context: dict[str, object]) -> list[str]:
    """Return missing fitment fields in the order we want to clarify them."""
    missing: list[str] = []
    if not fitment_context.get("vehicle"):
        missing.append("vehicle")
    if int(fitment_context.get("bike_count") or 0) < 1:
        missing.append("bike_count")
    if not fitment_context.get("bike_type"):
        missing.append("bike_type")
    if fitment_context.get("hitch") != "2-inch":
        missing.append("hitch")
    return missing


def _build_targeted_fitment_clarification(missing_fields: list[str]) -> str:
    """Ask exactly one targeted fitment question for the most critical missing field."""
    if "vehicle" in missing_fields:
        return "Got it — thanks. What year, make, and model is your vehicle?"
    if "bike_count" in missing_fields:
        return "Got it — thanks.\n\nTo recommend the right setup, how many bikes are you planning to carry?"
    if "bike_type" in missing_fields:
        return "Got it — thanks. Are you carrying a standard bike or an e-bike?"
    if "hitch" in missing_fields:
        return "Got it — thanks. Do you know if your vehicle has a 2-inch hitch installed?"
    return "Got it — thanks. Can you share one more detail about your vehicle, hitch, or bike setup?"


def _build_fitment_recommendation(fitment_context: dict[str, object]) -> str:
    """Generate a known-product fitment recommendation from complete structured data."""
    vehicle_name = str(fitment_context.get("vehicle") or "your vehicle")
    vehicle_year = fitment_context.get("vehicle_year")
    vehicle = f"your {vehicle_year} {vehicle_name}" if vehicle_year else f"your {vehicle_name}"
    bike = "one standard e-bike" if fitment_context.get("bike_type") == "ebike" else "your bike"

    return (
        f"Yes — based on {vehicle} with a 2-inch hitch and {bike}, this setup should work well for you.\n\n"
        "I would recommend the High Clearance Short Attachment Bar + Single All Bike configuration. "
        "That gives you a single-bike setup for the e-bike and provides added clearance behind the CR-V. "
        "Please still confirm your hitch and vehicle tongue-weight limits before loading."
    )


def _derive_escalation_reason(question: str, prior_intent: str | None) -> str:
    q = question.lower()
    if "warranty" in q or "liability" in q or "covered" in q or "void" in q:
        return "Warranty / liability question"
    if _looks_safety_critical(question):
        return "Safety-sensitive question"
    if prior_intent == "fitment compatibility" or _looks_fitment_question(question):
        return "Fitment unresolved"
    if "product" in q or "rack" in q:
        return "Unsupported product question"
    return "Insufficient context after retrieval"


def _question_topic(question: str, prior_intent: str | None = None) -> str:
    q = question.lower()
    if prior_intent == "fitment compatibility" or _looks_fitment_question(question):
        return "fitment"
    if "warranty" in q or "liability" in q or "covered" in q or "void" in q:
        return "warranty"
    if any(term in q for term in ("damage", "damages", "scratch", "scratches", "dent", "break", "broke", "safe", "safety", "battery")):
        return "safety_damage"
    if any(term in q for term in ("color", "colour", "dimension", "dimensions", "weight", "height", "width", "length", "spec", "come in")):
        return "product_info"
    return "general"


def _intent_to_topic(intent: str | None) -> str | None:
    """Normalize stored conversation intent labels to topic labels."""
    if not intent:
        return None
    if intent == "fitment compatibility":
        return "fitment"
    if intent in {"fitment", "warranty", "safety_damage", "product_info", "general"}:
        return intent
    return None


def _is_related_to_escalated_topic(new_question: str, escalated_question: str, prior_intent: str | None = None) -> bool:
    """Minimal deterministic topic check to keep unrelated questions fresh after escalation."""
    new_topic = _question_topic(new_question, prior_intent=None)
    old_topic = _question_topic(escalated_question, prior_intent=prior_intent)
    if new_topic != "general" and old_topic != "general" and new_topic != old_topic:
        return False

    new_terms = set(re.findall(r"[a-z0-9-]{4,}", new_question.lower()))
    old_terms = set(re.findall(r"[a-z0-9-]{4,}", escalated_question.lower()))
    stop_terms = {"what", "about", "does", "with", "rack", "gnarack", "radgnarack", "come"}
    overlap = (new_terms - stop_terms) & (old_terms - stop_terms)
    return bool(overlap) or new_topic == old_topic


def _build_escalation_summary(
    reason: str,
    original_question: str,
    latest_user_message: str,
    fitment_context: dict[str, object],
) -> str:
    if reason == "Fitment unresolved" and fitment_context:
        parts = []
        vehicle_year = fitment_context.get("vehicle_year")
        vehicle = fitment_context.get("vehicle")
        if vehicle:
            parts.append(f"{vehicle_year} {vehicle}".strip() if vehicle_year else str(vehicle))
        bike_count = fitment_context.get("bike_count")
        bike_type = fitment_context.get("bike_type")
        if bike_type:
            bike_text = f"{bike_count} {bike_type}" if bike_count else str(bike_type)
            parts.append(bike_text)
        hitch = fitment_context.get("hitch")
        if hitch:
            parts.append(f"{hitch} hitch")
        if parts:
            return "Customer asked about fitment with " + ", ".join(parts) + "."

    if reason == "Warranty / liability question":
        return f"Customer asked a warranty or liability question: {latest_user_message}"

    if reason == "Safety-sensitive question":
        return f"Customer asked a safety-sensitive question: {latest_user_message}"

    return f"Customer asked: {original_question}. Latest message: {latest_user_message}."


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    original_question = request.question.strip()
    question = original_question
    conversation_id = request.conversation_id

    if not original_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(
        "chat received conversation_id=%s original_question=%r",
        conversation_id,
        original_question,
    )

    active_escalation = (
        escalation_repo.get_active_by_conversation_id(conversation_id)
        if conversation_id
        else None
    )
    if active_escalation and not _is_related_to_escalated_topic(
        original_question,
        active_escalation.user_question,
    ):
        escalation_repo.close_active_for_conversation(conversation_id)
        logger.info(
            "chat closed stale escalation for unrelated new question conversation_id=%s escalation_id=%s original_question=%r escalated_question=%r",
            conversation_id,
            active_escalation.escalation_id,
            original_question,
            active_escalation.user_question,
        )

    # Check if this is a follow-up to a clarification
    enriched_question = question
    used_followup_context = False
    followup_detected = False
    prior_intent = None
    previous_turn_type = None
    prior_state = get_conversation_state(conversation_id) if conversation_id else None
    current_topic = _question_topic(original_question, prior_intent=None)
    previous_topic = None
    is_topic_switch = False
    if prior_state:
        prior_intent = prior_state.last_intent if prior_state else None
        previous_turn_type = prior_state.last_turn_type
        previous_topic = _intent_to_topic(prior_intent) or _question_topic(
            prior_state.last_question,
            prior_intent=None,
        )
        is_topic_switch = (
            previous_topic is not None
            and current_topic != "general"
            and previous_topic != "general"
            and current_topic != previous_topic
        )

    if is_topic_switch:
        if conversation_id:
            clear_conversation_state(conversation_id)
        prior_intent = current_topic
        enriched_question = original_question
        used_followup_context = False
        followup_detected = False
        logger.info(
            "chat topic switch detected conversation_id=%s previous_topic=%s current_topic=%s original_question=%r",
            conversation_id,
            previous_topic,
            current_topic,
            original_question,
        )

    if not is_topic_switch and not _looks_safety_critical(original_question):
        followup_detected = is_follow_up_to_clarification(conversation_id, original_question)

    if followup_detected:
        enriched = build_enriched_fitment_query(conversation_id, original_question)
        if enriched:
            enriched_question = enriched
            used_followup_context = True
            logger.info(
                "chat follow-up context applied conversation_id=%s original_question=%r used_followup_context=%s followup_detected=%s previous_turn_type=%s prior_intent=%s enriched_query_preview=%r",
                conversation_id,
                original_question,
                used_followup_context,
                followup_detected,
                previous_turn_type,
                prior_intent,
                enriched_question[:200],
            )

    result = answer_question(enriched_question)
    result_dict = result_to_dict(result)
    logger.info(
        "chat retrieval result conversation_id=%s original_question=%r enriched_query=%r followup_detected=%s result.status=%s",
        conversation_id,
        original_question,
        enriched_question if used_followup_context else None,
        followup_detected,
        result_dict.get("status"),
    )

    fitment_context = ""
    if prior_state and not is_topic_switch:
        fitment_context = " ".join(
            part for part in [prior_state.fitment_context, question, enriched_question]
            if part
        )
    else:
        fitment_context = question
    normalized_fitment_context = _normalize_fitment_context(
        fitment_context,
        current_question=question,
        previous_answer=prior_state.last_answer if prior_state and not is_topic_switch else "",
    )
    is_complete_fitment = _has_complete_fitment_data(normalized_fitment_context)
    missing_fitment_fields = _get_missing_fitment_fields(normalized_fitment_context)
    logger.info(
        "chat fitment context conversation_id=%s fitment_context=%s is_complete_fitment=%s missing_fitment_fields=%s",
        conversation_id,
        normalized_fitment_context,
        is_complete_fitment,
        missing_fitment_fields,
    )

    if (
        result_dict.get("status") == "insufficient_context"
        and result_dict.get("escalation_needed")
        and not followup_detected
        and _looks_fitment_question(original_question)
    ):
        result_dict["answer"] = _build_initial_fitment_clarification(original_question)
        result_dict["used_context"] = False
        result_dict["escalation_needed"] = False
        result_dict["status"] = "clarification_needed"
        result_dict["sources"] = []
        logger.info(
            "chat initial fitment clarification conversation_id=%s original_question=%r result.status=%s",
            conversation_id,
            original_question,
            result_dict.get("status"),
        )

    elif (
        result_dict.get("status") == "insufficient_context"
        and result_dict.get("escalation_needed")
        and prior_intent == "fitment compatibility"
        and followup_detected
        and is_complete_fitment
    ):
        result_dict["answer"] = _build_fitment_recommendation(normalized_fitment_context)
        result_dict["used_context"] = False
        result_dict["escalation_needed"] = False
        result_dict["status"] = "answered"
        result_dict["sources"] = []
        logger.info(
            "chat fitment recommendation generated conversation_id=%s used_followup_context=%s followup_detected=%s previous_turn_type=%s prior_intent=%s",
            conversation_id,
            used_followup_context,
            followup_detected,
            previous_turn_type,
            prior_intent,
        )

    elif (
        result_dict.get("status") == "insufficient_context"
        and result_dict.get("escalation_needed")
        and prior_intent == "fitment compatibility"
        and followup_detected
        and prior_state
        and not is_complete_fitment
    ):
        result_dict["answer"] = _build_targeted_fitment_clarification(missing_fitment_fields)
        result_dict["used_context"] = False
        result_dict["escalation_needed"] = False
        result_dict["status"] = "clarification_needed"
        result_dict["sources"] = []
        logger.info(
            "chat targeted fitment clarification conversation_id=%s used_followup_context=%s followup_detected=%s previous_turn_type=%s prior_intent=%s missing_fitment_fields=%s clarification_attempts=%s",
            conversation_id,
            used_followup_context,
            followup_detected,
            previous_turn_type,
            prior_intent,
            missing_fitment_fields,
            prior_state.clarification_attempts + 1,
        )

    logger.info(
        "chat decision conversation_id=%s original_question=%r enriched_query=%r used_followup_context=%s followup_detected=%s previous_turn_type=%s prior_intent=%s result.status=%s final_decision=%s",
        conversation_id,
        original_question,
        enriched_question if used_followup_context else None,
        used_followup_context,
        followup_detected,
        previous_turn_type,
        prior_intent,
        result.status,
        result_dict.get("status"),
    )
    
    # Store conversation state for potential follow-up
    if conversation_id:
        intent_for_state = (
            "fitment compatibility"
            if result_dict.get("status") == "clarification_needed"
            else current_topic
        )
        turn_type_for_state = (
            "clarification"
            if result_dict.get("status") == "clarification_needed"
            else None
        )
        previous_attempts = prior_state.clarification_attempts if prior_state else 0
        clarification_attempts = (
            previous_attempts + 1
            if result_dict.get("status") == "clarification_needed"
            else 0
        )
        previous_fitment_context = prior_state.fitment_context if prior_state else ""
        fitment_context_for_state = (
            " ".join(part for part in [previous_fitment_context, question] if part).strip()
            if result_dict.get("status") == "clarification_needed" or prior_intent == "fitment compatibility"
            else ""
        )
        set_conversation_state(
            conversation_id=conversation_id,
            question=original_question,  # Store original question, not enriched
            status=result_dict.get("status", ""),
            answer=result_dict.get("answer", ""),
            intent=intent_for_state,
            turn_type=turn_type_for_state,
            clarification_attempts=clarification_attempts,
            fitment_context=fitment_context_for_state,
        )

    # Check if escalation is needed
    if should_escalate(result_dict.get("status"), result_dict.get("escalation_needed")):
        current_state = get_conversation_state(conversation_id) if conversation_id else None
        escalation_reason = _derive_escalation_reason(original_question, prior_intent)
        original_for_escalation = (
            current_state.original_question
            if current_state and current_state.original_question
            else original_question
        )
        transcript = current_state.recent_turns[-8:] if current_state else []
        structured_context = normalized_fitment_context if normalized_fitment_context else None
        summary = _build_escalation_summary(
            reason=escalation_reason,
            original_question=original_for_escalation,
            latest_user_message=original_question,
            fitment_context=normalized_fitment_context,
        )
        # Create escalation asynchronously (don't block response)
        create_escalation(
            user_question=enriched_question if used_followup_context else original_question,
            conversation_id=request.conversation_id if hasattr(request, "conversation_id") else None,
            page_context=request.page_context if hasattr(request, "page_context") else None,
            source_url=request.source_url if hasattr(request, "source_url") else None,
            original_question=original_for_escalation,
            latest_user_message=original_question,
            conversation_summary=summary,
            structured_context=structured_context,
            escalation_reason=escalation_reason,
            recent_transcript=transcript,
        )

    return ChatResponse(**result_dict)
