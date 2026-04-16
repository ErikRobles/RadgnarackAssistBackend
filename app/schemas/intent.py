from pydantic import BaseModel
from typing import List

class IntentResult(BaseModel):
    intent: str  # faq | recommendation | lead | fallback
    confidence: float
    matched_rules: List[str]
    is_ambiguous: bool
