from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

async def call_llm(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Thin contract for calling LLM provider.
    Phase 1: Stub implementation.
    """
    # In a real implementation, this would call OpenAI/Gemini/etc.
    # For Phase 1 implementation without API keys, we log and return empty or mock if needed in tests.
    logger.debug(f"LLM Call - System: {system_prompt[:50]}... User: {user_prompt[:50]}...")
    raise NotImplementedError("LLM API not configured. Use mocks for testing.")

async def embed_query(text: str) -> List[float]:
    """Embedding generation contract (stub)."""
    return [0.0] * 1536  # Placeholder vector
