from sentence_transformers import SentenceTransformer
from app.domain.llm_providers.base_embedding_provider import BaseEmbeddingProvider


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> list:
        """
        Generate an embedding for the given text using all-MiniLM-L6-v2.
        Returns a list of floats.
        """
        return self.model.encode(text)
