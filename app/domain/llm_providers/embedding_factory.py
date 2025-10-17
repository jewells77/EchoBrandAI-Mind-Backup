from app.config import get_embedding_config
from app.domain.llm_providers.gemini_embedding_provider import (
    GoogleGenerativeAIEmbeddingProvider,
)


def get_embedding_provider(provider_name: str):
    """
    Factory to get the correct embedding provider.
    """
    if provider_name == get_embedding_config(provider=None)[0]:
        return GoogleGenerativeAIEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
