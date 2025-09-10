from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider


class ContentGeneratorAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.content_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a professional content creator who generates high-quality, engaging content.
You have exceptional reading comprehension and ALWAYS follow the exact requirements in the user's content request.

When generating content:
- PRECISELY follow any word count limits mentioned in the original request
- Use the appropriate format based on the request and context
- Match the tone, style, and voice requested
- Create engaging content tailored to the target audience
- Ensure factual accuracy and strategic keyword placement

You are skilled at interpreting instructions directly from natural language requests and delivering exactly what was asked for.
If a user asks for "50 word content" or "keep it under 100 words" or any similar instruction, you will honor that request precisely.

Your goal is to deliver content that follows the user's specifications WITHOUT needing additional processing or tracking.
""",
                ),
                (
                    "human",
                    """Theme: {theme}
Format: {format}
Brand Tone: {brand_tone}
Target Audience: {target_audience}
Original Request: {content_request}
Additional Context: {additional_context}

Please generate content that precisely follows the requirements in the original request.""",
                ),
            ]
        )

    async def generate_content(
        self,
        theme: str,
        format: str,
        content_request: str = "",
        brand_tone: Optional[str] = None,
        target_audience: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate draft content based on theme, format, and the original content request.

        Args:
            theme: The content theme or topic
            format: The content format (blog, social post, video script, etc.)
            content_request: Original content request containing any specific instructions
            brand_tone: Optional brand voice/tone to match
            target_audience: Optional target audience description
            additional_context: Optional additional context or requirements

        Returns:
            Dict containing the content draft
        """
        # Format prompt with inputs
        formatted_prompt = self.content_prompt.format_messages(
            theme=theme,
            format=format,
            brand_tone=brand_tone or "Professional and engaging",
            target_audience=target_audience or "General audience",
            content_request=content_request or "No specific requirements provided",
            additional_context=additional_context or "",
        )

        # Get response from LLM
        response = await self.llm.generate(formatted_prompt)

        return {
            "draft": response.content,
            "metadata": {
                "theme": theme,
                "format": format,
            },
        }
