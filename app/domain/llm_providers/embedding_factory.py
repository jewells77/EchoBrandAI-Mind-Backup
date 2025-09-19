from app.domain.llm_providers.hf_embedding_provider import HuggingFaceEmbeddingProvider

# from app.domain.llm_providers.openai_embedding_provider import OpenAIEmbeddingProvider
# from app.domain.llm_providers.gemini_embedding_provider import GeminiEmbeddingProvider


def get_embedding_provider(provider_name: str):
    """
    Factory to get the correct embedding provider.
    """
    if provider_name == "huggingface":
        return HuggingFaceEmbeddingProvider()
    # elif provider_name == "openai":
    #     return OpenAIEmbeddingProvider()
    # elif provider_name == "gemini":
    #     return GeminiEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
