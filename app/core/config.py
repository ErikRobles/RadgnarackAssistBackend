from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "RadgnarackAssist"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Pinecone Settings
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None # Optional, depending on Pinecone tier
    PINECONE_NAMESPACE: str = "default"

    # Retrieval Settings
    DEFAULT_TOP_K: int = 3
    MIN_RELEVANCE_SCORE: float = 0.70

    # LLM Settings
    ENABLE_LLM: bool = False
    LLM_MODEL: str = "gpt-4o"  # Placeholder for future provider integration

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
