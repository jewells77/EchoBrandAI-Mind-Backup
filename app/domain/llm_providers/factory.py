from enum import Enum
from typing import Dict, Optional, Type

from app.config import settings
from app.domain.llm_providers.base import BaseLLMProvider
from app.domain.llm_providers.openai_provider import OpenAIProvider


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"


class LLMProviderFactory:
    """Factory class for creating LLM providers."""

    _providers: Dict[LLMProviderType, Type[BaseLLMProvider]] = {
        LLMProviderType.OPENAI: OpenAIProvider,
    }

    @classmethod
    def get_provider(
        cls, provider_type: LLMProviderType = LLMProviderType.OPENAI, **kwargs
    ) -> BaseLLMProvider:
        """
        Get an instance of the requested LLM provider.

        Args:
            provider_type: Type of provider to create (defaults to OPENAI)
            **kwargs: Additional parameters to pass to the provider constructor

        Returns:
            BaseLLMProvider: An instance of the requested provider

        Raises:
            ValueError: If the requested provider type is not supported
        """
        provider_class = cls._providers.get(provider_type)

        if not provider_class:
            raise ValueError(f"Unsupported LLM provider type: {provider_type}")

        return provider_class(**kwargs)

    @classmethod
    def register_provider(
        cls, provider_type: str, provider_class: Type[BaseLLMProvider]
    ) -> None:
        """
        Register a new provider type.

        Args:
            provider_type: Type identifier for the provider
            provider_class: The provider class to register
        """
        cls._providers[provider_type] = provider_class


def create_llm_provider(**kwargs) -> BaseLLMProvider:
    """
    Create an LLM provider based on application settings.

    Args:
        **kwargs: Additional parameters to pass to the provider constructor

    Returns:
        BaseLLMProvider: A configured LLM provider
    """
    # Default to OpenAI provider
    provider_type = LLMProviderType.OPENAI

    return LLMProviderFactory.get_provider(provider_type, **kwargs)
