from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    @abstractmethod
    def embed_query(self, text: str) -> list:
        pass

    @abstractmethod
    def embed_documents(self, texts: list) -> list:
        pass
