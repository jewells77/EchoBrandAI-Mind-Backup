from typing import List, Dict, Any
import json
from typing_extensions import TypedDict, Annotated

from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider


class BrandPersonaProfile(TypedDict):
    """Brand persona profile with extracted tone, audience and positioning."""

    brand_tone: Annotated[
        str,
        ...,
        "The voice and emotional quality of the brand's communication",
    ]
    target_audience: Annotated[
        str,
        ...,
        "Detailed description of the ideal customer or audience",
    ]
    unique_positioning: Annotated[
        str,
        ...,
        "What makes this brand different from competitors",
    ]
    keywords: Annotated[List[str], ..., "Key phrases that define the brand identity"]
    visual_elements: Annotated[
        str,
        ...,
        "Recommended visual elements that align with brand identity",
    ]


class BrandDNAAnalyzerAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a brand analyst expert who extracts the core DNA of a brand. 
Analyze the provided brand details to identify:

1. Brand tone - The voice and emotional quality of the brand's communication
2. Target audience - Detailed description of the ideal customer
3. Unique positioning - What makes this brand stand out from competitors
4. Keywords - Key phrases that define the brand identity
5. Visual elements - Recommended visual elements that align with brand identity

Return your analysis as a structured JSON object.
""",
                ),
                (
                    "human",
                    """Brand Details: 
{brand_details}

Extract the brand DNA and provide a structured profile.""",
                ),
            ]
        )

    async def analyze(self, brand_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze brand details to extract brand tone, target audience, and unique positioning.
        Returns a structured brand persona profile as JSON.

        Args:
            brand_details: Dictionary containing brand information

        Returns:
            Dictionary containing the brand persona profile
        """
        result = await self.llm.generate(
            prompt=self.prompt,
            input={
                "brand_details": json.dumps(brand_details, indent=2),
            },
            output_schema=BrandPersonaProfile,
        )
        return result
