import uuid
from app.schemas.lead import LeadResult

async def process_lead(user_text: str) -> LeadResult:
    """LEADS: Purely data-oriented lead capture logic."""
    normalized = user_text.strip()
    
    # 1. EMPTY / MALFORMED CHECK
    if not normalized or len(normalized) < 5:
        return LeadResult(
            status="insufficient_info",
            reference_id="",
            extracted_data={}
        )
    
    # 2. CAPTURED
    # Generate a deterministic/stubbed ID for Phase 1
    # In Phase 1 we just return a new UUID to simulate tracking
    ref_id = str(uuid.uuid4())[:8].upper()
    
    return LeadResult(
        status="captured",
        reference_id=ref_id,
        extracted_data={
            "original_request": normalized
        }
    )
