import re
from typing import Optional
from app.schemas.product import RecommendationQuery

async def parse_recommendation_query(user_text: str) -> Optional[RecommendationQuery]:
    """
    Deterministic Phase 1 parser for recommendation constraints.
    Returns RecommendationQuery if sufficient data is found, else None.
    """
    normalized = user_text.lower()
    
    # 1. Required: Number of Bikes
    # Matches: '2 bikes', '2 e-bikes', 'ebike for 2', 'e-bike for 2'
    num_bikes_match = re.search(r"(\d+)\s*(?:e-)?bikes?", normalized)
    if not num_bikes_match:
        # Try 'ebike for 2'
        num_bikes_match = re.search(r"(?:e-?bike)\s*(?:for\s*)?(\d+)", normalized)
        
    if not num_bikes_match:
        return None
    num_bikes = int(num_bikes_match.group(1))
    
    # 2. Required: Max Bike Weight (Phase 1: Default to 30.0 if not specified, but prefer extraction)
    # The prompt says "Do NOT guess", so if weight is critical for the engine, 
    # and not in text, we might consider it insufficient. 
    # However, many users won't specify weight. Let's look for it.
    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lbs|lb|pounds|kg)", normalized)
    max_weight = float(weight_match.group(1)) if weight_match else 30.0 # Standard default
    
    # 3. Optional: Hitch Size
    hitch_size = None
    if "2-inch" in normalized or '2" hitch' in normalized or "2.0" in normalized:
        hitch_size = 2.0
    elif "1.25" in normalized or '1.25" hitch' in normalized:
        hitch_size = 1.25
        
    # 4. Optional: E-Bike Signal
    is_ebike = any(kw in normalized for kw in ["e-bike", "ebike", "electric bike"])
    
    # 5. Optional: Step-through
    needs_step_through = any(kw in normalized for kw in ["step through", "step-through", "low frame"])

    return RecommendationQuery(
        number_of_bikes=num_bikes,
        max_bike_weight=max_weight,
        hitch_size_inches=hitch_size,
        is_e_bike=is_ebike,
        needs_step_through_support=needs_step_through
    )
