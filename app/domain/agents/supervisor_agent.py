from typing import Any, Dict
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.domain.llm_providers.base import BaseLLMProvider


class SupervisorDecision(TypedDict):
    """Structured output from the supervisor agent."""

    next_agent: str


class SupervisorAgent:
    """
    Decides if image generation is needed for the current content state.
    Returns a dict: { 'call_image_agent': bool, 'image_prompt': str, 'supervisor_msg': str }
    """

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a supervisor agent. Your only task is to decide
whether the user wants to generate an image based on their query.

Respond with ONLY one word:
- 'image_agent' if the image agent should be called
- 'finalize' if no image generation is needed

Do not add explanations or any other text.""",
                ),
                ("human", "{user_qurey}"),
            ]
        )

    async def decide(self, user_qurey: str) -> str:
        result = await self.llm.generate(
            prompt=self.prompt,
            input={"user_qurey": user_qurey},
            output_schema=SupervisorDecision,
        )
        return result["next_agent"]
