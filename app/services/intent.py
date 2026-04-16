import re
from typing import List, Dict, Set
from app.schemas.intent import IntentResult

# Rule IDs and Regex Patterns
RULES: Dict[str, Dict[str, str]] = {
    "LEAD": {
        "lead_buy": r"\bbuy\b|\border\b|\bpurchase\b",
        "lead_contact": r"\bquote\b|\bsales\b|\bdemo\b|\bcontact\b|\brepresentative\b"
    },
    "RECOMMENDATION": {
        "rec_suggest": r"\bsuggest\b|\brecommend\b|\brecommendation\b|\bwhich rack\b|\bbest for\b|\brack\b",
        "rec_choice": r"\bchoice for\b|\bcomparative\b"
    },
    "FAQ": {
        "faq_specs": r"\bhitch\b|\bweight limit\b|\bload capacity\b|\bfitting\b",
        "faq_price": r"\bhow much\b|\bcost\b|\bprice\b",
        "faq_support": r"\bwarranty\b|\bmanual\b"
    }
}

PRIORITY = ["LEAD", "RECOMMENDATION", "FAQ"]

async def classify_intent(user_text: str) -> IntentResult:
    """Deterministic intent classification based on hierarchical rules."""
    
    # Normalization: lower, trim, collapse multiple spaces
    normalized = " ".join(user_text.lower().strip().split())
    
    if not normalized:
        return _fallback_result()

    matched_categories: Set[str] = set()
    all_matched_rules: List[str] = []
    
    # Rule Matching
    for category, rules in RULES.items():
        category_matched = False
        for rule_id, pattern in rules.items():
            if re.search(pattern, normalized):
                all_matched_rules.append(rule_id)
                category_matched = True
        
        if category_matched:
            matched_categories.add(category)

    if not matched_categories:
        return _fallback_result()

    # Ambiguity Detection
    is_ambiguous = len(matched_categories) > 1
    
    # Priority Resolution
    final_intent = "fallback"
    for category in PRIORITY:
        if category in matched_categories:
            final_intent = category.lower()
            break
            
    return IntentResult(
        intent=final_intent,
        confidence=1.0,
        matched_rules=all_matched_rules,
        is_ambiguous=is_ambiguous
    )

def _fallback_result() -> IntentResult:
    return IntentResult(
        intent="fallback",
        confidence=0.0,
        matched_rules=[],
        is_ambiguous=False
    )
