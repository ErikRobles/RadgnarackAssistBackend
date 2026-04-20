import anyio
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
from app.core.exceptions import ConfigurationError, RetrievalError, ServiceUnavailableError

# Global client and index references for thin adapter
_pc: Optional[Pinecone] = None
_index: Optional[Any] = None

def _get_client():
    global _pc, _index
    if not _pc:
        if not settings.PINECONE_API_KEY:
            raise ConfigurationError("Missing PINECONE_API_KEY in configuration.")
        if not settings.PINECONE_INDEX_NAME:
            raise ConfigurationError("Missing PINECONE_INDEX_NAME in configuration.")
            
        try:
            _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            _index = _pc.Index(settings.PINECONE_INDEX_NAME)
        except Exception as e:
            raise ServiceUnavailableError(f"Failed to initialize Pinecone: {str(e)}")
            
    return _index

async def upsert_vectors(vectors: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
    """
    Thin Pinecone upsert contract.
    Expects List[Dict] with: 'id', 'values', 'metadata'
    """
    try:
        index = _get_client()
        # Run sync SDK call in thread pool for async compatibility
        await anyio.to_thread.run_sync(
            lambda: index.upsert(
                vectors=vectors,
                namespace=namespace or settings.PINECONE_NAMESPACE
            )
        )
    except (ConfigurationError, ServiceUnavailableError):
        raise
    except Exception as e:
        raise RetrievalError(f"Error during Pinecone upsert: {str(e)}")

async def query_index(
    vector: List[float], 
    filters: Optional[Dict[str, Any]] = None, 
    top_k: int = 3,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Thin wrapper around Pinecone query for async safety."""
    
    try:
        # Run synchronous SDK call in a thread pool for async compatibility
        index = _get_client()
        
        raw_response = await anyio.to_thread.run_sync(
            lambda: index.query(
                vector=vector,
                filter=filters,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace or settings.PINECONE_NAMESPACE
            )
        )
        
        # Normalize and return thin raw match dictionaries
        raw_matches = raw_response.get("matches", []) if isinstance(raw_response, dict) else getattr(raw_response, "matches", [])
        matches = []
        for match in raw_matches:
            if isinstance(match, dict):
                matches.append({
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "metadata": match.get("metadata", {})
                })
            else:
                matches.append({
                    "id": getattr(match, "id", None),
                    "score": getattr(match, "score", None),
                    "metadata": getattr(match, "metadata", {}) or {}
                })
        return matches
        
    except (ConfigurationError, ServiceUnavailableError):
        raise
    except Exception as e:
        raise RetrievalError(f"Error during Pinecone query: {str(e)}")
