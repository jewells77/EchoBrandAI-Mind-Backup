from typing import Dict, Any, List
from typing_extensions import TypedDict, Annotated
from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider
from app.api.v1.schemas.common import flatten_dict


class RefinedContent(TypedDict):
    """Structured refined content output."""

    final_content: Annotated[str, ..., "Publication-ready refined content"]
    character_count: Annotated[int, ..., "Character count of the final content"]


class ContentRefinerAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.refine_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a professional content editor and refiner who polishes draft content into a publication-ready piece.

==============================
     STRICT RULES & POLICIES
==============================
1. Always preserve word count constraints if they exist.
2. Refine the draft without altering its intended meaning or purpose.
3. Enhance clarity, flow, readability, and engagement while maintaining brand tone.
4. Always include or refine a CTA.
5. Ensure grammar, spelling, and SEO optimization where applicable.
6. Produce exactly ONE refined post per request (no splitting or multiple outputs).
7. Never include unsafe, offensive, or misleading language.
8. Only use the provided context: draft content, brand guidelines, tone, target audience, keywords, and original request.

==============================
     OBJECTIVE
==============================
Return a structured JSON object with one refined, polished, brand-aligned, 
and publication-ready piece of content that strictly follows the original request.
""",
                ),
                (
                    "human",
                    """Draft Content:
{draft_content}

Brand Guidelines:
{brand_guidelines}

Tone Requirements:
{tone}

Target Audience:
{target_audience}

Original Request:
{user_qurey}

SEO Keywords (if applicable):
{keywords}

Please refine this content to be publication-ready while strictly adhering to all requirements 
from the original request, especially any word count limits.""",
                ),
            ]
        )

    async def refine_content(
        self, draft_content: str, guidelines: Dict[str, Any], user_qurey: str = ""
    ) -> Dict[str, Any]:
        """
        Polish language, ensure brand consistency, add CTA.
        Returns final ready-to-publish content.

        Args:
            draft_content: The draft content to refine
            guidelines: Dict containing refining guidelines including:
                - brand_guidelines: Brand style/voice guidelines
                - tone: Desired tone for the content
                - target_audience: Target audience description
                - keywords: SEO keywords to include (optional)
            user_qurey: Original user query to understand user requirements

        Returns:
            Dict containing the final refined content
        """
        # Extract guidelines
        brand_guidelines = guidelines.get("brand_guidelines", "")
        tone = guidelines.get("tone", "")
        target_audience = guidelines.get("target_audience", "")
        keywords = guidelines.get("keywords", [])

        # Format keywords as string if they're provided as a list
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)

        result = await self.llm.generate(
            prompt=self.refine_prompt,
            input={
                "draft_content": flatten_dict(draft_content),
                "brand_guidelines": flatten_dict(brand_guidelines),
                "tone": flatten_dict(tone),
                "target_audience": flatten_dict(target_audience),
                "user_qurey": flatten_dict(user_qurey)
                or "No specific requirements provided",
                "keywords": flatten_dict(keywords),
            },
            output_schema=RefinedContent,
        )

        # Robustness: fall back if the LLM omits optional fields
        final_content: str = result.get("final_content", "")
        character_count: int = (
            result.get("character_count")
            if isinstance(result.get("character_count"), int)
            else len(final_content)
        )

        return {
            "final_content": final_content,
            "metadata": {
                "character_count": character_count,
            },
        }
