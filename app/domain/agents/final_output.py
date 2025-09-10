from typing import Dict, Any
from typing_extensions import TypedDict, Annotated

from langchain.prompts import ChatPromptTemplate

from app.domain.llm_providers.base import BaseLLMProvider


class FinalContent(TypedDict):
    """Final output containing a strong title and the final content body."""

    title: Annotated[str, ..., "A compelling, concise title for the content"]
    content: Annotated[str, ..., "The final, publication-ready content body"]


class FinalOutputAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a senior editor. Based on the provided context, produce a strong title and the final content body.
Return ONLY the requested fields as a structured object.""",
                ),
                (
                    "human",
                    """Brand Profile:
{brand_profile}

Content Strategy:
{content_strategy}

Draft Content:
{draft_content}

Refined Content:
{refined_content}

Produce a compelling title and the content.""",
                ),
            ]
        )

    async def finalize(
        self,
        brand_profile: Dict[str, Any],
        content_strategy: Dict[str, Any],
        content_draft: Dict[str, Any],
        final_content: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = await self.llm.generate(
            prompt=self.prompt,
            input={
                "brand_profile": str(brand_profile),
                "content_strategy": str(content_strategy),
                "draft_content": str(content_draft.get("draft", "")),
                "refined_content": str(final_content.get("final_content", "")),
            },
            output_schema=FinalContent,
        )
        return result
