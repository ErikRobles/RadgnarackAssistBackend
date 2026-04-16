from pydantic import BaseModel, Field
from typing import List, Optional, Any

class ProductProfile(BaseModel):
    product_id: str
    name: str
    max_bikes: int
    max_weight_per_bike: float
    supported_hitch_sizes: List[float]
    max_tire_width: float
    supports_step_through: bool
    is_ebike_rated: bool
    extension_clearance_inches: float
    base_price: float

class RecommendationQuery(BaseModel):
    number_of_bikes: int
    max_bike_weight: float
    hitch_size_inches: Optional[float] = None
    tire_width_inches: Optional[float] = None
    is_e_bike: bool = False
    needs_step_through_support: bool = False
    has_spare_tire_on_rear: bool = False

class ExclusionDetail(BaseModel):
    product_id: str
    reason: str

class RecommendationResult(BaseModel):
    matches: List[ProductProfile]
    excluded: List[ExclusionDetail]
    is_sufficient: bool
    engine_metadata: Dict[str, str] = Field(default_factory=lambda: {"version": "1.0.0-deterministic"})

from typing import Dict
