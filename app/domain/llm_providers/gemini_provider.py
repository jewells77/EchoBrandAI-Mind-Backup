from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage
from app.config import settings
from app.domain.llm_providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.kwargs = kwargs
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            **self.kwargs,
        )

    async def generate(self, *, messages: List[BaseMessage], **kwargs) -> Any:
        # messages: List[BaseMessage] (e.g., HumanMessage with text/image content)
        return await self.client.ainvoke(messages)

    async def stream(
        self, messages: List[Union[Dict[str, str], BaseMessage]], **kwargs
    ) -> AsyncGenerator[AIMessage, None]:
        # Streaming not implemented for Gemini in this example
        raise NotImplementedError("Streaming not supported for GeminiProvider.")

    def with_structured_output(self, schema: Any, **kwargs) -> Any:
        # Not implemented for Gemini in this example
        raise NotImplementedError("Structured output not supported for GeminiProvider.")

    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
        }
