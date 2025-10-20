from typing import Any, Dict, List, Annotated
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.domain.llm_providers.base import BaseLLMProvider


class BrandAnalysis(TypedDict):
    """Structured insights and summary of the brand's own materials."""

    brand_summary: Annotated[
        str,
        ...,
        "Comprehensive summary of brand content and positioning.",
    ]


class BrandAnalysisAgent:
    """
    Agent for analyzing and summarizing a brand's own materials.
    Extracts key identity, values, tone, and market positioning from raw text data.
    """

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a structured brand summarization agent.

You will receive an array of strings. Each element represents raw marketing content, website copy, brand documents, or public communications from a single brand.

Your task:
1. Carefully review all text elements to extract **key brand insights**.
2. Produce a **concise and cohesive summary (300–400 words)** that covers:
   - **Brand Overview / Identity** — mission, vision, purpose, or story
   - **Products or Services** — main offerings or categories
   - **Tone & Messaging Style** — how the brand communicates (formal, friendly, innovative, luxury, etc.)
   - **Target Audience** — who the brand speaks to
   - **Core Values / Differentiators** — what makes the brand unique
   - **Customer Experience / Reputation** (if evident)
   - **Geographic Focus / Market Presence** (if applicable)
3. Keep the tone **neutral and factual**, avoiding promotional adjectives.
4. Remove repetitive or redundant statements.
5. Output should be formatted in **clear Markdown**, with headings and bullet points.

Your goal is to generate a complete, well-structured profile of the brand — suitable for strategic analysis or positioning work.
                    """,
                ),
                (
                    "human",
                    """brand_data: {brand_data}""",
                ),
            ]
        )

    async def analyze(self, brand_data: List[str]) -> str:
        """
        Analyze the brand's content and produce a structured summary.
        """
        prompt_vars = {"brand_data": brand_data}
        result = await self.llm.generate(
            prompt=self.prompt,
            input=prompt_vars,
            output_schema=BrandAnalysis,
        )
        return result["brand_summary"]
