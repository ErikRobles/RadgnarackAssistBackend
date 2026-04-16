from typing import List, Optional, Any, Dict
import logging
from app.schemas.retrieval import KnowledgeChunk
from app.schemas.product import ProductProfile, RecommendationQuery
from app.schemas.lead import LeadResult
from app.adapters import llm_provider
from app.core.config import settings
from app.services import safety

logger = logging.getLogger(__name__)

async def generate_faq_response(
    documents: List[KnowledgeChunk],
    user_query: str
) -> str:
    """Generates FAQ response, optionally enhanced by LLM."""
    # 1. Deterministic base
    if not documents:
        return ""
    
    deterministic_base = "Based on our records: " + " ".join([doc.text for doc in documents])
    
    # 2. LLM Enhancement (Optional)
    if settings.ENABLE_LLM:
        try:
            system_prompt = (
                "You are a technical sales assistant for bike racks. "
                "Use ONLY the provided evidence. Do not invent information. "
                "If the answer is not in the evidence, say the information is not available."
            )
            evidence = "\n".join([f"- {doc.text}" for doc in documents])
            user_prompt = f"Evidence:\n{evidence}\n\nUser Question: {user_query}"
            
            candidate = await llm_provider.call_llm(system_prompt, user_prompt)
            
            if candidate and candidate.strip():
                # Internal safety check for LLM output
                context = {"mode": "faq", "documents": documents}
                if await safety.validate_response(candidate, context):
                    return candidate.strip()
                else:
                    logger.warning("LLM FAQ output rejected by safety. Falling back to deterministic.")
        except Exception as e:
            logger.error(f"LLM FAQ generation failed: {str(e)}. Falling back to deterministic.")
            
    return deterministic_base

async def generate_recommendation_response(
    matches: List[ProductProfile],
    query: RecommendationQuery
) -> str:
    """Generates recommendation explanation, optionally enhanced by LLM."""
    # 1. Deterministic base
    if not matches:
        return ""
    
    parts = ["I recommend the following racks based on your needs:"]
    for product in matches:
        desc = (
            f"- {product.name}: Supports {product.max_bikes} bikes "
            f"(up to {product.max_weight_per_bike} lbs each). "
            f"Fits {', '.join([str(h) for h in product.supported_hitch_sizes])}-inch hitches."
        )
        if product.is_ebike_rated:
            desc += " It is e-bike rated."
        parts.append(desc)
    
    deterministic_base = "\n".join(parts)
    
    # 2. LLM Enhancement (Optional)
    if settings.ENABLE_LLM:
        try:
            system_prompt = (
                "You are a technical sales assistant for bike racks. "
                "Explain the recommended products based ONLY on the provided product data. "
                "Do not invent specifications. Do not mention products not in the list."
            )
            # Serialize product data for LLM
            evidence = "\n".join([
                f"- {p.name}: {p.max_bikes} bikes, {p.max_weight_per_bike}lb limit, "
                f"hitch: {p.supported_hitch_sizes}, ebike: {p.is_ebike_rated}"
                for p in matches
            ])
            user_prompt = f"Recommended Products:\n{evidence}\n\nFormatting request: Provide a helpful summary explanation."
            
            candidate = await llm_provider.call_llm(system_prompt, user_prompt)
            
            if candidate and candidate.strip():
                context = {"mode": "recommendation", "matches": matches}
                if await safety.validate_response(candidate, context):
                    return candidate.strip()
                else:
                    logger.warning("LLM Recommendation output rejected by safety. Falling back to deterministic.")
        except Exception as e:
            logger.error(f"LLM Recommendation generation failed: {str(e)}. Falling back to deterministic.")
            
    return deterministic_base

async def generate_lead_response(
    lead_result: LeadResult
) -> str:
    """Deterministic acknowledgment based on lead result status (LLM not used for leads)."""
    if lead_result.status == "insufficient_info":
        return "I can help connect you with our sales team. Could you please provide a bit more detail about what you are looking for?"
        
    msg = "Thank you! Your information has been received. A representative will contact you shortly."
    if lead_result.reference_id:
        msg += f" (Reference ID: {lead_result.reference_id})"
    return msg

async def generate_explanation(structured_data: Any, mode: str) -> str:
    """Adapter for Orchestrator."""
    if mode == "faq":
        docs = structured_data if isinstance(structured_data, list) else structured_data.documents
        return await generate_faq_response(docs, "")
    elif mode == "recommendation":
        prods = structured_data if isinstance(structured_data, list) else structured_data.matches
        # We don't have the original query object here easily without changing orchestrator, 
        # but recommendation mode still works with matches.
        return await generate_recommendation_response(prods, None)
    elif mode == "lead":
        return await generate_lead_response(structured_data)
    return ""
