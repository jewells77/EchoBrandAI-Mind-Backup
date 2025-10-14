from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.domain.llm_providers.base import BaseLLMProvider
from app.config import settings


class ValidationResult(TypedDict):
    is_validate: bool
    message: str


class ValidationAgent:
    def __init__(
        self,
        llm: BaseLLMProvider,
    ):
        self.llm = llm
        self.brand_detail_guide_url = settings.BRAND_DETAIL_GUIDE_URL
        self.competitor_site_guide_url = settings.COMPETITOR_SITE_GUIDE_URL
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
You are a helpful and polite onboarding assistant. Respond with friendly, encouraging, and natural language in every message, never sounding robotic.

You will be provided with two boolean flags:
- is_brand_detail: True if the user's brand/company details are provided; False if missing.
- is_competitor_site: True if one or more competitor URLs have been provided; False if missing.

When telling the user how to provide missing brand details, direct them to this link: {settings.BRAND_DETAIL_GUIDE_URL}
When telling the user how to provide competitor site URLs, direct them to this link: {settings.COMPETITOR_SITE_GUIDE_URL}

If ANY required detail is missing (either value is False), set `is_validate` to false and give concise, clear, but conversational guidance on how to provide the missing information and where to go.
If BOTH are present (True/True), set `is_validate` to true and confirm in a friendly way that all requirements are met.
Always output exactly and only JSON: {{{{"is_validate": bool, "message": str}}}}.
Never explain your own reasoning—just return the result formatted for the user.
""",
                ),
                (
                    "human",
                    """is_brand_detail: {is_brand_detail}
is_competitor_site: {is_competitor_site}
user_query: {user_query}""",
                ),
            ]
        )

    async def decide(
        self, is_brand_detail: bool, is_competitor_site: bool, user_query: str
    ) -> ValidationResult:
        input_vars = {
            "is_brand_detail": is_brand_detail,
            "is_competitor_site": is_competitor_site,
            "user_query": user_query,
        }
        result = await self.llm.generate(
            prompt=self.prompt,
            input=input_vars,
            output_schema=ValidationResult,
        )
        return result
