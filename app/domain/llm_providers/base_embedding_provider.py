from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> list:
        pass
