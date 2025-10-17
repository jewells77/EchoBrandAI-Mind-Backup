from typing import Optional, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.domain.llm_providers.base_embedding_provider import BaseEmbeddingProvider
from app.config import settings, get_embedding_config
from app.api.exceptions import APIError


class GoogleGenerativeAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self, model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs
    ):
        provider = settings.GOOGLE_EMBEDDING_PROVIDER
        _, _model, _dim = get_embedding_config(provider=provider, model=model_name)
        self.model_name = _model
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.kwargs = kwargs
        self.model = GoogleGenerativeAIEmbeddings(
            model=self.model_name, google_api_key=self.api_key, **self.kwargs
        )

    def embed_query(self, text: str) -> list:
        """
        Generate an embedding for a single query (text) using Gemini embeddings.
        Returns a list of floats.
        """
        try:
            return self.model.embed_query(text)
        except Exception as e:
            raise APIError(f"Failed to generate embedding: {e}", status_code=500)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts (documents).
        Args:
            texts: A list of text strings to embed.
        Returns:
            A list of embedding vectors.
        """
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            raise APIError(
                f"Failed to generate document embeddings: {e}", status_code=500
            )
