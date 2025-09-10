from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider


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
{content_request}

SEO Keywords (if applicable):
{keywords}

Please refine this content to be publication-ready while strictly adhering to all requirements 
from the original request, especially any word count limits.""",
                ),
            ]
        )

    async def refine_content(
        self, draft_content: str, guidelines: Dict[str, Any], content_request: str = ""
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
            content_request: Original content request to understand user requirements

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

        # Format prompt with inputs
        formatted_prompt = self.refine_prompt.format_messages(
            draft_content=draft_content,
            brand_guidelines=brand_guidelines,
            tone=tone,
            target_audience=target_audience,
            content_request=content_request or "No specific requirements provided",
            keywords=keywords,
        )

        # Get response from LLM
        response = await self.llm.generate(formatted_prompt)

        # Determine content type/format from the refined content
        content_format = self._determine_content_format(response.content)

        return {
            "final_content": response.content,
            "metadata": {
                "format": content_format,
                "character_count": len(response.content),
            },
        }

    def _determine_content_format(self, content: str) -> str:
        """
        Determine the format of the content based on structure and markers.

        Args:
            content: The content to analyze

        Returns:
            String indicating the detected format
        """
        # Simple format detection based on content markers
        content_lower = content.lower()

        if (
            content.startswith("#")
            or "<h1>" in content_lower
            or "<h2>" in content_lower
        ):
            return "blog_post"
        elif "subject:" in content_lower or "dear" in content_lower[:100]:
            return "email"
        elif "fade in:" in content_lower or "scene" in content_lower[:100]:
            return "video_script"
        elif len(content) < 500 and ("#" in content or "@" in content):
            return "social_media_post"
        elif "introduction" in content_lower and "conclusion" in content_lower:
            return "article"
        else:
            return "general_content"
