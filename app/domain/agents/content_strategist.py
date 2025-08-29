from typing import Dict, Any, List
import json
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.domain.llm_providers.base import BaseLLMProvider


class ContentStrategy(BaseModel):
    """Content strategy document with titles, formats, angles and recommendations."""

    titles: List[str] = Field(description="Potential content titles or topics")
    formats: List[str] = Field(
        description="Recommended content formats (blog, video, infographic, etc.)"
    )
    angles: List[str] = Field(
        description="Creative angles or approaches for the content"
    )
    hashtags: List[str] = Field(
        description="Recommended hashtags for social media promotion"
    )
    target_platforms: List[str] = Field(
        description="Best platforms for content distribution"
    )
    cta_suggestions: List[str] = Field(
        description="Call-to-action suggestions for the content"
    )


class ContentStrategistAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ContentStrategy)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a creative content strategist who develops content strategies aligned with brand identity.
Analyze the brand profile, competitor insights, and content request to develop a comprehensive content strategy including:

1. Content titles/topics - Specific, engaging content ideas
2. Content formats - Best formats for delivery (blog, video, carousel, etc.)
3. Creative angles - Unique approaches that will resonate with the audience
4. Hashtags - Strategic hashtags for social media distribution
5. Target platforms - Best channels to reach the target audience
6. CTA suggestions - Effective calls to action aligned with content goals

Return your strategy as a structured JSON object.
{format_instructions}
""",
                ),
                (
                    "human",
                    """Brand Profile: 
{brand_profile}

Competitor Insights:
{competitor_insights}

Content Request:
{content_request}

Develop a content strategy based on this information.""",
                ),
            ]
        )

    async def suggest_strategy(
        self,
        brand_profile: Dict[str, Any],
        competitor_insights: Dict[str, Any],
        content_request: str,
    ) -> Dict[str, Any]:
        """
        Suggest content themes, formats, and creative angles.
        Returns a strategy document.
        """
        # Format prompt with inputs and parser instructions
        formatted_prompt = self.prompt.format_messages(
            brand_profile=json.dumps(brand_profile, indent=2),
            competitor_insights=json.dumps(competitor_insights, indent=2),
            content_request=content_request,
            format_instructions=self.parser.get_format_instructions(),
        )

        # Get response from LLM
        response = await self.llm.generate(formatted_prompt)

        try:
            # Parse the response into our Pydantic model
            parsed_response = self.parser.parse(response.content)
            # Convert to dict for return
            return parsed_response.model_dump()
        except Exception as e:
            # Fallback in case parsing fails
            return {
                "titles": ["Could not generate titles due to parsing error"],
                "formats": ["Could not generate formats due to parsing error"],
                "angles": ["Could not generate angles due to parsing error"],
                "hashtags": [],
                "target_platforms": [],
                "cta_suggestions": [],
                "error": str(e),
            }
