"""
Agent: PreQuestionAgent (Perplexity-integrated)
----------------------------------------------
Uses the perplexity_search tool to fetch live brand insights, then
creates three personalised pre-chat questions:
    - Two customer-style
    - One content-creator-style
"""

from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.domain.llm_providers.base import BaseLLMProvider
from app.domain.tools.perplexity_search import perplexity_search
from fastapi import HTTPException
import traceback # Import for printing full traceback on error

class PreQuestionResult(TypedDict):
    questions: list[str]


class PreQuestionAgent:
    """Autonomous agent that reasons, calls Perplexity, and writes creative brand questions."""

    def __init__(self, llm: BaseLLMProvider):
        # We access the underlying LangChain client (.client) to bind the tools.
        # using perplexity search tool for agent
        self.llm = llm.client.bind_tools([perplexity_search])

        # System prompt for agent
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
f"""
You are a creative assistant who crafts engaging, **brand-specific** pre-chat question suggestions that guide users toward meaningful, answerable topics for our advanced AI content system.

You have access to the `perplexity_search` tool.

Do NOT output the tool call arguments as your final answer.
Each question must naturally include the brand-related keyword and be phrased to seek current, ongoing, or time-relevant information about the brand — ensuring retrieve recent or trending insights without sounding repetitive or robotic.
Questions should be easy to understand, friendly in tone, and sound like something a real end-user would naturally ask — not overly robotic or technical. & each time new tone and style of question.
Your final and ONLY output must be the JSON structure defined in step 7.

Tone and style:
- Write short, natural, fan-like questions (15–25 words) that sound curious, conversational, and practical.
- Must include **brand-related keywords** naturally.
- Avoid yes/no phrasing or repetitive wording.
- Make sure each question can lead to an actionable or informative response from the system (e.g., content creation, brand info, post writing, trend insight, or feature highlight).

Follow this workflow carefully:
1. Read the full `brand_details`.
2. Identify brand nature (product, service, technology, fashion, agency, etc.) by analyzing words in `brand_details`.
3. Generate one comprehensive search query to explore the brand’s latest trends, news, and updates (01/01/2024–12/31/2025).
4. Call the `perplexity_search` tool once with that query.
5. Review the observation (titles + snippets).
6. If the tool call fails or returns no useful info, rely on general brand understanding.
7. Using both brand_detais and the insights from the tool, craft **exactly three pre-chat questions** designed to trigger valuable AI responses:
    - **Q1 (Awareness Post)**: Ask to create a post or caption that highlights the brand’s products, trends, or offers — suitable for platforms like Instagram, Twitter, or Facebook etc.
    - **Q2 (Creative/Promotional Content)**: Ask to generate creative content — e.g., a caption, full social media post, or storytelling idea related to the brand’s vibe or message etc.
    - **Q3 (Strategic/Collaborative Content)**: Ask for professional, brand-use content — like campaign ideas, LinkedIn posts, brand collabs, or marketing text ready for publication etc.
    - Each question must use a different angle (no repetition).
8. Output the final result strictly as JSON:
{{{{ "questions": ["question1", "question2", "question3"] }}}}
"""
            ),
            ("human", "brand_details: {brand_details}")
        ])
            
    async def generate(self, brand_details: str) -> PreQuestionResult:
        """
        Input:
            brand_details (str): Basic info about the brand.
        Behavior:
            - Dynamically creates brand-specific queries.
            - Calls perplexity_search once to fetch live info.
            - Generates 3 creative pre-chat questions.
        """
        # Define the LangChain Runnable chain
        chain = self.prompt | self.llm.with_structured_output(PreQuestionResult)

        try:
            # Use chain.ainvoke
            result = await chain.ainvoke({"brand_details": brand_details})

            # This safely handles the output from chain.ainvoke
            result_dict = dict(result)

            if "questions" in result_dict:
                return result_dict["questions"]
            else:
                # If 'questions' is missing, raise a specific error to capture the bad output
                raise KeyError(f"LLM output missing 'questions' key. Full result: {result_dict}")

        except Exception as e:
            # CRITICAL DEBUGGING: Log the full error
            print(f"CRITICAL LLM FAILURE. Error: {e.__class__.__name__}: {e}")
            traceback.print_exc()

            # Raise the clean HTTP exception for the API endpoint
            raise HTTPException(
                status_code=500,
                detail={
                    "error_type": "LLM_GENERATION_FAILED",
                    "message": "The AI agent failed to generate questions due to an internal service error (API key, network, or invalid LLM response schema).",
                    "reason": f"Encountered error: {e}"
                }
            )