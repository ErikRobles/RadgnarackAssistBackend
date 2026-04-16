from typing import List, Optional
from app.schemas.product import (
    ProductProfile, 
    RecommendationQuery, 
    ExclusionDetail, 
    RecommendationResult
)

async def get_product_recommendations(
    query: RecommendationQuery, 
    products: List[ProductProfile]
) -> RecommendationResult:
    """Deterministic rule-based product recommendation engine."""
    matches: List[ProductProfile] = []
    excluded: List[ExclusionDetail] = []
    
    for p in products:
        # 1. Fail-Closed Check: Verify required attributes exist
        if p.max_bikes is None or p.max_weight_per_bike is None or p.supported_hitch_sizes is None:
            excluded.append(ExclusionDetail(product_id=p.product_id, reason="Missing required product attributes (Fail-Closed)"))
            continue
            
        # 2. Hard Exclusion Rules
        
        # Bike count check
        if query.number_of_bikes > p.max_bikes:
            excluded.append(ExclusionDetail(product_id=p.product_id, reason="Bike count exceeds max capacity"))
            continue
            
        # Weight check
        if query.max_bike_weight > p.max_weight_per_bike:
            excluded.append(ExclusionDetail(product_id=p.product_id, reason="Bike weight exceeds max per-bike weight limit"))
            continue
            
        # Hitch compatibility
        if query.hitch_size_inches and query.hitch_size_inches not in p.supported_hitch_sizes:
            excluded.append(ExclusionDetail(product_id=p.product_id, reason="Unsupported hitch size"))
            continue
            
        # E-bike compatibility
        if query.is_e_bike and not p.is_ebike_rated:
            excluded.append(ExclusionDetail(product_id=p.product_id, reason="Not rated for e-bikes"))
            continue
            
        # Step-through support
        if query.needs_step_through_support and not p.supports_step_through:
            excluded.append(ExclusionDetail(product_id=p.product_id, reason="Step-through support required but not available"))
            continue
            
        # Tire width check
        if query.tire_width_inches and query.tire_width_inches > p.max_tire_width:
            excluded.append(ExclusionDetail(product_id=p.product_id, reason="Tire width exceeds product max"))
            continue
            
        # If all rules pass, it's a match
        matches.append(p)
        
    # 3. Deterministic Ranking
    # Rule 1: Exact bike count match preferred
    # Rule 2: Higher weight margin preferred (as secondary tie-breaker)
    matches.sort(
        key=lambda x: (
            abs(x.max_bikes - query.number_of_bikes), # Smaller difference first
            -(x.max_weight_per_bike - query.max_bike_weight) # Larger margin first
        )
    )
    
    return RecommendationResult(
        matches=matches,
        excluded=excluded,
        is_sufficient=len(matches) > 0
    )
