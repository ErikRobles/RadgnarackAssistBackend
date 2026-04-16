from pydantic import BaseModel
from typing import Dict, Any

class LeadResult(BaseModel):
    status: str  # captured | insufficient_info
    reference_id: str
    extracted_data: Dict[str, Any]
