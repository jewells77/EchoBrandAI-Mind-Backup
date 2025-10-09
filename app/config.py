# path: app/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional, ClassVar, List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "EcoBrandAI"
    DEBUG: bool = False

    # LLM Provider settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: Optional[int] = None
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: Optional[str] = os.getenv("LANGCHAIN_PROJECT", "")

    # MongoDB settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "ecobrandai"
    MONGODB_CHECKPOINT_COLLECTION: str = "langgraph_checkpoints"
    MONGODB_WRITES_COLLECTION: str = "langgraph_writes"

    # Vectorstore settings
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_WEBSITE_CONTENT_COLLECTION: ClassVar[str] = "website-content"
    QDRANT_COLLECTIONS: ClassVar[List[Dict[str, list]]] = [
        {
            "name": QDRANT_WEBSITE_CONTENT_COLLECTION,
            "required_payload": ["url", "user_id", "text"],
            "optional_payload": ["timestamp"],
        }
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
