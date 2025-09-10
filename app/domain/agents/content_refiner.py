from typing import Dict, Any, List
from typing_extensions import TypedDict, Annotated
from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider


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
                    """You are a professional content editor and refiner who specializes in polishing content to match brand guidelines.
You have exceptional reading comprehension and ALWAYS follow the exact requirements in the user's original content request.

Your task is to take draft content and refine it by:
1. Correcting grammar, spelling, and punctuation
2. Improving flow, structure, and readability
3. Ensuring consistent brand voice and tone
4. Optimizing for SEO with strategic keyword placement
5. Adding or improving calls-to-action (CTAs)
6. Ensuring the content matches the target audience
7. Removing any redundancies or unnecessary sections
8. Enhancing clarity and impact

CRITICAL: You MUST preserve any word count limits from the original request. If the user asked for "50 word content" 
or similar, your refined output must strictly adhere to that limit without needing additional tracking or processing.

The refined content should be publication-ready and maintain the original format while making these improvements.
\n
Default behavior:
- Unless the original request clearly and explicitly asks for MULTIPLE pieces with a specific number (e.g., "3 posts", "two tweets", "a 5-part series"), ensure the output is EXACTLY ONE cohesive piece.
- Do NOT split into multiple posts or sections like "Post 1", "Post 2" unless the request explicitly specifies a count.

Return your output as a structured JSON object.
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
                "draft_content": draft_content,
                "brand_guidelines": brand_guidelines,
                "tone": tone,
                "target_audience": target_audience,
                "user_qurey": user_qurey or "No specific requirements provided",
                "keywords": keywords,
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
