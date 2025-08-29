from typing import List, Dict, Any
import json

from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.domain.llm_providers.base import BaseLLMProvider


class BrandPersonaProfile(BaseModel):
    """Brand persona profile with extracted tone, audience and positioning."""

    brand_tone: str = Field(
        description="The voice and emotional quality of the brand's communication"
    )
    target_audience: str = Field(
        description="Detailed description of the ideal customer or audience"
    )
    unique_positioning: str = Field(
        description="What makes this brand different from competitors"
    )
    keywords: List[str] = Field(
        description="Key phrases that define the brand identity"
    )
    visual_elements: str = Field(
        description="Recommended visual elements that align with brand identity"
    )


class BrandDNAAnalyzerAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=BrandPersonaProfile)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a brand analyst expert who extracts the core DNA of a brand. 
Analyze the provided brand details and competitor information to identify:

1. Brand tone - The voice and emotional quality of the brand's communication
2. Target audience - Detailed description of the ideal customer
3. Unique positioning - What makes this brand stand out from competitors
4. Keywords - Key phrases that define the brand identity
5. Visual elements - Recommended visual elements that align with brand identity

Return your analysis as a structured JSON object.
{format_instructions}
""",
                ),
                (
                    "human",
                    """Brand Details: 
{brand_details}

Competitors:
{competitors}

Extract the brand DNA and provide a structured profile.""",
                ),
            ]
        )

    async def analyze(
        self, brand_details: Dict[str, Any], competitors: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze brand details and competitors to extract brand tone, target audience, and unique positioning.
        Returns a structured brand persona profile as JSON.

        Args:
            brand_details: Dictionary containing brand information
            competitors: Optional list of competitor URLs or names (default: None)

        Returns:
            Dictionary containing the brand persona profile
        """
        # Format competitors list into a string
        competitors = competitors or []
        competitors_text = "\n".join([f"- {comp}" for comp in competitors])

        # Format prompt with user inputs and parser instructions
        formatted_prompt = self.prompt.format_messages(
            brand_details=json.dumps(brand_details, indent=2),
            competitors=competitors_text,
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
                "brand_tone": "Could not extract brand tone due to parsing error",
                "target_audience": "Could not extract target audience due to parsing error",
                "unique_positioning": "Could not extract unique positioning due to parsing error",
                "keywords": [],
                "visual_elements": "Could not extract visual elements due to parsing error",
                "error": str(e),
            }
