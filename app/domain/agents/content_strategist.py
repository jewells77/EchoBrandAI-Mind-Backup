from typing import Dict, Any, List
import json
from typing_extensions import TypedDict, Annotated
from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider


class ContentStrategy(TypedDict):
    """Content strategy document with titles and recommendations."""

    titles: Annotated[List[str], ..., "Potential content titles or topics"]
    hashtags: Annotated[
        List[str],
        ...,
        "Recommended hashtags for social media promotion",
    ]
    cta_suggestions: Annotated[
        List[str],
        ...,
        "Call-to-action suggestions for the content",
    ]


class ContentStrategistAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a creative content strategist who develops content strategies aligned with brand identity.
Analyze the brand profile, competitor insights, and content request to develop a concise content strategy including:

1. Content titles/topics - Specific, engaging content ideas
2. Hashtags - Strategic hashtags for social media distribution
3. CTA suggestions - Effective calls to action aligned with content goals

Do not include a formats list. Return your strategy as a structured JSON object.
""",
                ),
                (
                    "human",
                    """Brand Profile: 
{brand_profile}

Competitor Insights:
{competitor_insights}

Content Request:
{user_qurey}

Develop a content strategy based on this information.""",
                ),
            ]
        )

    async def suggest_strategy(
        self,
        brand_profile: Dict[str, Any],
        competitor_insights: Dict[str, Any],
        user_qurey: str,
    ) -> Dict[str, Any]:
        """
        Suggest content themes.
        Returns a strategy document.
        """

        result = await self.llm.generate(
            prompt=self.prompt,
            input={
                "brand_profile": json.dumps(brand_profile, indent=2),
                "competitor_insights": json.dumps(competitor_insights, indent=2),
                "user_qurey": user_qurey,
            },
            output_schema=ContentStrategy,
        )
        return result
