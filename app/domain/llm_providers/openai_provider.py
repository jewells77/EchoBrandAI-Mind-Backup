import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import AIMessage, BaseMessage

from app.config import settings
from app.domain.llm_providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation using LangChain."""

    def __init__(
        self,
        model_name: str = settings.OPENAI_MODEL_NAME,
        temperature: float = settings.OPENAI_TEMPERATURE,
        max_tokens: Optional[int] = settings.OPENAI_MAX_TOKENS,
        api_key: Optional[str] = settings.OPENAI_API_KEY,
        **kwargs
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model_name: The OpenAI model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key
            **kwargs: Additional parameters to pass to ChatOpenAI
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.kwargs = kwargs

        # Initialize the ChatOpenAI client
        self.client = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            streaming=False,
            **self.kwargs
        )

    async def generate(
        self, messages: List[Union[Dict[str, str], BaseMessage]], **kwargs
    ) -> AIMessage:
        """
        Generate a response from OpenAI based on input messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            AIMessage: The generated response
        """
        normalized_messages = self._normalize_messages(messages)

        # Apply any runtime configuration overrides
        client = self._configure_client(streaming=False, **kwargs)

        # Get response from OpenAI
        response = await client.ainvoke(normalized_messages)
        return response

    async def stream(
        self, messages: List[Union[Dict[str, str], BaseMessage]], **kwargs
    ) -> AsyncGenerator[AIMessage, None]:
        """
        Stream a response from OpenAI based on input messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            AsyncGenerator[AIMessage, None]: Generator yielding chunks of the response
        """
        normalized_messages = self._normalize_messages(messages)

        # Set up streaming callback handler
        callback_handler = AsyncIteratorCallbackHandler()

        # Configure client for streaming
        client = self._configure_client(
            streaming=True, callbacks=[callback_handler], **kwargs
        )

        # Start generating in the background
        task = asyncio.create_task(client.ainvoke(normalized_messages))

        # Stream the response
        async for chunk in callback_handler.aiter():
            yield AIMessage(content=chunk)

        # Ensure the task completes
        await task

    def _configure_client(self, streaming: bool = False, **kwargs) -> ChatOpenAI:
        """
        Configure the OpenAI client with runtime parameters.

        Args:
            streaming: Whether to enable streaming
            **kwargs: Additional parameters to override

        Returns:
            ChatOpenAI: Configured client
        """
        # Start with default parameters
        config = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
            "streaming": streaming,
            **self.kwargs,
        }

        # Override with any runtime parameters
        config.update(kwargs)

        # Create a new client with the updated configuration
        return ChatOpenAI(**config)
