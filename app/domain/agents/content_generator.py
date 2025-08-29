from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider


class ContentGeneratorAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a professional content creator who generates high-quality, engaging content.
Generate content based on the provided theme and format. Create content that is:

- Engaging and tailored to the target audience
- Structured appropriately for the specified format
- On-brand with the tone and style provided
- Factually accurate and well-researched
- SEO-friendly with strategic keyword placement

Follow the specific format instructions:
- For blog posts: Include title, headers, and body content with proper structure
- For social media: Create a post with appropriate hashtags and call-to-action
- For video scripts: Include opening hook, main sections, and closing call-to-action
- For email campaigns: Include subject line, greeting, body, and closing
- For product descriptions: Include features, benefits, and specifications
- For infographics: Include title, section headers, and content for each section

Always consider the platform-specific best practices for the requested format.
""",
                ),
                (
                    "human",
                    """Theme: {theme}
Format: {format}
Brand Tone: {brand_tone}
Target Audience: {target_audience}
Additional Context: {additional_context}

Please generate a draft for this content.""",
                ),
            ]
        )

    async def generate_content(
        self,
        theme: str,
        format: str,
        brand_tone: Optional[str] = None,
        target_audience: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate draft content based on theme and format.

        Args:
            theme: The content theme or topic
            format: The content format (blog, social post, video script, etc.)
            brand_tone: Optional brand voice/tone to match
            target_audience: Optional target audience description
            additional_context: Optional additional context or requirements

        Returns:
            Dict containing the content draft
        """
        # Format prompt with inputs
        formatted_prompt = self.prompt.format_messages(
            theme=theme,
            format=format,
            brand_tone=brand_tone or "Professional and engaging",
            target_audience=target_audience or "General audience",
            additional_context=additional_context or "",
        )

        # Get response from LLM
        response = await self.llm.generate(formatted_prompt)

        return {
            "draft": response.content,
            "metadata": {"theme": theme, "format": format},
        }
