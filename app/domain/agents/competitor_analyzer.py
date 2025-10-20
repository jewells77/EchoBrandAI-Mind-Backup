from typing import Any, Dict, Literal, Annotated, List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.domain.llm_providers.base import BaseLLMProvider


class CompetitorInsights(TypedDict):
    """Structured insights from competitor content analysis."""

    competitor_insights: Annotated[
        str,
        ...,
        "summary of competitor insights",
    ]


class CompetitorIntelligenceAgent:
    """
    Central policy and routing for the agentic workflow.
    Passes explicit key state to the LLM, for maximal clarity and guidance.
    """

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a precise and comprehensive summarization agent.

You will receive an array of strings. Each element represents raw website or marketing content from one or more companies or brands. 
The content may contain mixed data from different businesses, promotional language, offers, contact info, or testimonials.

Your task:
1. Read all array elements carefully and identify **every unique, meaningful detail**.
2. Produce a **single cohesive summary (400–500 words maximum)** that:
   - Captures all distinct insights, features, and offerings mentioned across all brands.
   - Merges redundant or repetitive content.
   - Preserves uniqueness without missing any specific idea or data point.
3. Organize the summary clearly using logical sections such as:
   - Overview / Brand Description  
   - Services & Offerings  
   - Unique Selling Points (USPs)  
   - Customer Experience or Testimonials (if any)  
   - Pricing / Offers (if mentioned)  
   - Locations & Contact Details (if available)
4. Use a **neutral, factual tone** — remove promotional adjectives like “best,” “amazing,” etc.
5. Avoid listing each brand separately unless the text explicitly distinguishes them.
6. The result should be concise, coherent, and within **400–500 words**, prioritizing completeness over style.

Your output must be formatted in **clear Markdown** with headings and bullet points where appropriate.

""",
                ),
                (
                    "human",
                    """competitors_data: {competitors_data}""",
                ),
            ]
        )

    async def analyze(self, competitors_data: List[str]) -> str:
        prompt_vars = {
            "competitors_data": competitors_data,
        }
        result = await self.llm.generate(
            prompt=self.prompt,
            input=prompt_vars,
            output_schema=CompetitorInsights,
        )
        return result["competitor_insights"]
