from typing import Any, Dict, Optional, Type, List, Union, Sequence
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import BaseTool
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
        **kwargs,
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
        self.provider = "openai"
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
            **self.kwargs,
        )

    async def generate(
        self,
        *,
        prompt: Any,
        input: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Any:
        """
        Generate a response from OpenAI using prompt chaining.

        Args:
            prompt: Runnable/prompt to chain with the client
            input: Optional dict of variables for the prompt (defaults to {})
            output_schema: Optional Pydantic model for structured output
            **kwargs: Extra parameters for runtime client config

        Returns:
            Any: The generated response (AIMessage for plain chat, or structured type)
        """
        if prompt is None:
            raise ValueError("'prompt' is required for generate().")

        prompt_input = input or {}

        # Apply runtime configuration (streaming, temperature, etc.)
        client = self._configure_client(streaming=False, **kwargs)

        # Switch between normal and structured client
        if output_schema:
            llm = client.with_structured_output(output_schema)
        else:
            llm = client

        chain = prompt | llm
        response = await chain.ainvoke(prompt_input)
        return response

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

    def with_structured_output(self, schema: Any, **kwargs) -> Any:
        """
        Return a runnable configured to produce structured output as per the given schema.

        This leverages LangChain's ChatOpenAI.with_structured_output, which supports
        TypedDict + Annotated and Pydantic models.
        """
        client = self._configure_client(streaming=False, **kwargs)
        return client.with_structured_output(schema)

    def create_agent(
        self,
        *,
        tools: list[BaseTool] | None = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Any] = None,
        middleware: Optional[Sequence[AgentMiddleware]] = None,
        **kwargs,
    ):
        """
        Create a LangGraph agent configured with this provider's model.
        Args:
            tools: List of tools/callables specific to this agent.
            system_prompt: The agent’s role/instructions.
            response_format: Optional structured output schema.
        """
        client = self._configure_client(streaming=False, **kwargs)
        # Dynamically build arguments, excluding None values
        agent_args = {
            "model": client,
            "system_prompt": system_prompt,
        }

        if tools:  # include only if non-empty
            agent_args["tools"] = tools

        if response_format is not None:
            agent_args["response_format"] = response_format

        if middleware:
            agent_args["middleware"] = middleware

        agent = create_agent(**agent_args)
        return agent

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the OpenAI provider.

        Returns:
            Dict[str, Any]: Information about the provider including name, model, etc.
        """
        return {
            "provider": "openai",
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": (
                self.max_tokens if self.max_tokens else "Not set (using model default)"
            ),
            "api_key_configured": bool(self.api_key),
        }
