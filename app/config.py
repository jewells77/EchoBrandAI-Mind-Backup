# path: app/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional, ClassVar, List, Dict, Union
from dotenv import load_dotenv
from pydantic import Field, field_validator

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "EcoBrandAI"
    DEBUG: bool = False

    # LLM Provider settings
    SHOW_WORKFLOW_GRAPH: bool = False
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: Optional[int] = None
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: Optional[str] = os.getenv("LANGCHAIN_PROJECT", "")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")

    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
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
    # Azure Blob Storage settings
    AZURE_BLOB_CONNECTION_STRING: Optional[str] = os.getenv(
        "AZURE_BLOB_CONNECTION_STRING", ""
    )
    AZURE_BLOB_CONTAINER: Optional[str] = os.getenv("AZURE_BLOB_CONTAINER", "")
    AZURE_BLOB_AI_GENERATED_IMAGES_PATH: str = "ai-generated/images"
    AZURE_BLOB_AI_GENERATED_VIDEOS_PATH: str = "ai-generated/videos"
    # Azure Blob accepted domains (comma separated string or list)
    ALLOWED_AZURE_BLOB_BASE_URLS: Union[str, list[str]] = Field(default="")

    @field_validator("ALLOWED_AZURE_BLOB_BASE_URLS", mode="after")
    @classmethod
    def parse_allowed_blob_urls(cls, v):
        if isinstance(v, list):
            return [url.rstrip("/") for url in v]
        if not v.strip():
            return []
        return [url.strip().rstrip("/") for url in v.split(",") if url.strip()]

    # CORS settings
    ALLOWED_ORIGINS: Union[str, list[str]] = Field(default="")

    @field_validator("ALLOWED_ORIGINS", mode="after")
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse comma-separated string into a list of origins."""
        if not v.strip():
            return [""]
        return [origin.strip() for origin in v.split(",") if origin.strip()] or [""]

    BRAND_DETAIL_GUIDE_URL: str = os.getenv(
        "BRAND_DETAIL_GUIDE_URL", "https://www.no-url.com"
    )
    COMPETITOR_SITE_GUIDE_URL: str = os.getenv(
        "COMPETITOR_SITE_GUIDE_URL", "https://www.no-url.com"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
