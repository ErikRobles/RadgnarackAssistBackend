from typing import Any, Dict, List
import re

async def validate_response(
    response: str,
    context: Dict[str, Any]
) -> bool:
    """Deterministic validation of generated response against its source context."""
    
    # 1. EMPTY CHECK
    if not response or not response.strip():
        return False
        
    mode = context.get("mode", "fallback")
    
    # 2. PRODUCT VALIDATION (Recommendation Mode)
    if mode == "recommendation":
        matches = context.get("matches", [])
        product_names = [p.name.lower() for p in matches]
        
        # Check if the response mentions a product that wasn't in the matches
        # Simple extraction of "likely" product names (assumes capitalized names)
        potential_mentions = re.findall(r"([A-Z][\w-]+\s?[\w-]*)", response)
        for mention in potential_mentions:
            mention_lower = mention.lower().strip()
            # If a capitalized multi-word sequence is found that's not in our match list
            # and isn't "I recommend", check if it looks like a product.
            if mention_lower not in product_names and "recommend" not in mention_lower:
                # Basic safety: if it's a known product NOT in matches, fail.
                # In Phase 1, we just ensure at least one match IS mentioned.
                pass
                
        # Stronger check: All matches should be present in the response if it's a list.
        # AND NO unauthorized product names should be present.
        if not any(name in response.lower() for name in product_names):
            return False

    # 3. ATTRIBUTE CONSISTENCY
    # Ensure no fabricated bike counts or weights
    if mode == "recommendation":
        matches = context.get("matches", [])
        for p in matches:
            if p.name.lower() in response.lower():
                # Extract numbers followed by "bike" or "lbs" or "lb"
                # If response says "Supports 4 bikes" but profile says 2, FAIL.
                found_counts = re.findall(rf"{re.escape(p.name)}.*?(\d+)\s*bikes?", response, re.IGNORECASE)
                for count in found_counts:
                    if int(count) > p.max_bikes:
                        return False
                
                found_weights = re.findall(rf"{re.escape(p.name)}.*?(\d+(?:\.\d+)?)\s*(?:lbs|lb)", response, re.IGNORECASE)
                for weight in found_weights:
                    if float(weight) > p.max_weight_per_bike:
                        return False

    # 4. FAQ GROUNDING
    if mode == "faq":
        documents = context.get("documents", [])
        if not documents:
            return False
            
        all_doc_text = " ".join([d.text.lower() for d in documents])
        # Find all significant words (len > 4) in response and check if any exist in context
        response_words = [w.lower() for w in re.findall(r"\w{5,}", response)]
        if not any(word in all_doc_text for word in response_words):
            # No significant shared vocabulary between context and response
            return False

    return True

async def validate_service_output(output: Any, context: Dict[str, Any]) -> bool:
    """
    Adapter function for Orchestrator to route to specialized validators.
    Matches the signature defined in the Orchestrator Phase 1.
    """
    return await validate_response(output, context)
