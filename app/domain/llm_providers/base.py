from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain.schema import AIMessage, BaseMessage, HumanMessage


class BaseLLMProvider(ABC):
    """Base class for LLM providers used in the application."""

    @abstractmethod
    async def generate(self, **kwargs) -> Any:
        """
        Generate a response from the LLM.

        Args:
            **kwargs: Implementation-specific parameters (e.g., prompt, input, schema)

        Returns:
            Any: The generated response (could be AIMessage, dict, string, etc.)
        """
        pass

    @abstractmethod
    async def stream(
        self, messages: List[Union[Dict[str, str], BaseMessage]], **kwargs
    ) -> AsyncGenerator[AIMessage, None]:
        """
        Stream a response from the LLM based on input messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            AsyncGenerator[AIMessage, None]: Generator yielding chunks of the response
        """
        pass

    @abstractmethod
    def with_structured_output(self, schema: Any, **kwargs) -> Any:
        """
        Return an LLM runnable configured to produce structured output matching the given schema.

        Args:
            schema: A TypedDict, Pydantic model, or JSON schema-compatible type
            **kwargs: Optional overrides for the underlying client configuration

        Returns:
            A runnable supporting .invoke/.ainvoke that yields the structured object
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM provider.

        Returns:
            Dict[str, Any]: Information about the provider including name, model, etc.
        """
        pass

    def _normalize_messages(
        self, messages: List[Union[Dict[str, str], BaseMessage]]
    ) -> List[BaseMessage]:
        """
        Normalize messages to LangChain's BaseMessage format.

        Args:
            messages: List of messages which may be dictionaries or BaseMessage instances

        Returns:
            List[BaseMessage]: List of normalized messages
        """
        normalized_messages = []

        for message in messages:
            if isinstance(message, BaseMessage):
                normalized_messages.append(message)
            elif isinstance(message, dict):
                if message.get("role") == "user":
                    normalized_messages.append(
                        HumanMessage(content=message.get("content", ""))
                    )
                elif message.get("role") == "ai":
                    normalized_messages.append(
                        AIMessage(content=message.get("content", ""))
                    )
                # Add more types as needed

        return normalized_messages
