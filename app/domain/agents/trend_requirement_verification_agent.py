from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.domain.llm_providers.base import BaseLLMProvider
#from datetime import datetime


class TrendRequirementVerificationResult(TypedDict):
    """Output schema for trend verification agent."""
    is_trend_needed: bool
    message: str


class TrendRequirementVerificationAgent:
    """
    Decides if a user's query requires external or trend-based exploration.
    It doesn't explore the internet itself — just verifies if it's needed.
    """

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""
You are a reasoning assistant that verifies whether a user query requires external or trend-based (internet) exploration.

Your task:
1. Understand the intent of `user_query`.
2. Decide if it relies on current or real-world info.

Rules:
- Set `is_trend_needed` = **true** if the query depends on:
  - current events, industry updates, real-time data, "latest", "nowadays", "in 2025", etc. In other words, internet exploration is needed.
- Set `is_trend_needed` = **false** if it can be answered using reasoning, general knowledge, or creativity. In other words, no internet exploration is needed.

Output must always be JSON:
{{{{ "is_trend_needed": bool}}}}

Example:
- true → if Trend analysis or internet exploration is equired for this query.(e.g., latest trends, current events, real-time data, locations updates, etc.)
- false → if Static query; no internet exploration needed.(chaning tone, style, lenght, etc.)
"""
            ),
            ("human", "user_query: {user_query}"),
        ])

    async def verify(self, user_query: str) -> TrendRequirementVerificationResult:
        """Checks if trend/internet lookup is needed."""
        input_vars = {"user_query": user_query}
        result = await self.llm.generate(
            prompt=self.prompt,
            input=input_vars,
            output_schema=TrendRequirementVerificationResult,
        )
        return result