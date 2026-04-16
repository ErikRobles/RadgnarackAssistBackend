# app/services/fallback.py

FALLBACK_MESSAGES = {
    "fallback_intent": "I'm sorry, I'm not sure how to help with that. Could you try rephrasing your request?",
    "no_results": "I couldn't find any products or information that match your specific request at this time.",
    "safety_failed": "I apologize, but I cannot provide a response for that request due to safety or grounding concerns.",
    "insufficient_recommendation_input": "I need a bit more information to make a safe recommendation. Could you specify how many bikes you have and any hitch requirements?",
    "system_error": "I'm sorry, a system error occurred. Please try again later."
}

async def get_fallback_response(reason: str) -> str:
    """ERROR HANDLING: Safe failure responses based on deterministic reason codes."""
    return FALLBACK_MESSAGES.get(reason, FALLBACK_MESSAGES["system_error"])
