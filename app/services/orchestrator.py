from typing import Optional, List
import logging
import re
from app.schemas.chat import ChatResponse
from app.core.products import PRODUCT_CATALOG
from app.services import (
    intent,
    retrieval,
    recommendation,
    recommendation_parser,
    lead_handler,
    response_service,
    safety,
    fallback
)

logger = logging.getLogger(__name__)

async def handle_message(user_text: str) -> ChatResponse:
    """End-to-end coordination: Intent -> Domain -> Response -> Safety -> Result."""
    try:
        # STEP 1: INTENT
        intent_result = await intent.classify_intent(user_text)
        
        # STEP 2: FALLBACK INTENT
        if intent_result.intent == "fallback":
            return await _trigger_fallback("fallback_intent")

        final_msg = ""
        context = {"mode": intent_result.intent}
        
        # STEP 3: ROUTING
        if intent_result.intent == "faq":
            # Retrieval
            ret_result = await retrieval.retrieve_faq_context(user_text)
            if not ret_result.is_sufficient:
                return await _trigger_fallback("no_results")
            
            # Response Generation
            final_msg = await response_service.generate_faq_response(ret_result.documents, user_text)
            context["documents"] = ret_result.documents

        elif intent_result.intent == "recommendation":
            # Deterministic Extraction via dedicated service
            query = await recommendation_parser.parse_recommendation_query(user_text)
            if not query:
                return await _trigger_fallback("insufficient_recommendation_input")
            
            # Engine
            rec_result = await recommendation.get_product_recommendations(query, PRODUCT_CATALOG)
            if not rec_result.is_sufficient:
                return await _trigger_fallback("no_results")
            
            # Response Generation
            final_msg = await response_service.generate_recommendation_response(rec_result.matches, query)
            context["matches"] = rec_result.matches

        elif intent_result.intent == "lead":
            # Lead Handler returns LeadResult
            lead_result = await lead_handler.process_lead(user_text)
            
            # Response Generation consumes LeadResult
            final_msg = await response_service.generate_lead_response(lead_result)
            context["lead_status"] = lead_result.status

        # STEP 4: SAFETY CHECK
        is_safe = await safety.validate_response(final_msg, context)
        if not is_safe:
            return await _trigger_fallback("safety_failed")

        # STEP 5: SUCCESS RESPONSE
        return ChatResponse(message=final_msg, intent=intent_result.intent)

    except Exception as e:
        logger.error(f"System Error in Orchestrator: {str(e)}", exc_info=True)
        return await _trigger_fallback("system_error")

async def _trigger_fallback(reason: str) -> ChatResponse:
    msg = await fallback.get_fallback_response(reason)
    return ChatResponse(message=msg, intent="fallback")
