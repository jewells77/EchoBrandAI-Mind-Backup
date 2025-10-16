from typing import Any, Dict, Literal
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.domain.llm_providers.base import BaseLLMProvider


class SupervisorDecision(TypedDict):
    """Structured output from the supervisor agent."""

    next_agent: Literal[
        "validation_agent",
        "image_agent",
        "final_output_agent",
        "end",
    ]


class SupervisorAgent:
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
                    """You are the SUPERVISOR agent — the main controller of an agentic LangGraph workflow.

Your job is to analyze the user query and decide which agent should act next.

Always remember:
- Only you decide which agent should execute next.
- Each agent returns control back to you after completion.
- You keep the workflow cyclic until explicitly ended.

Available agents:
1. validation_agent
2. image_agent
3. final_output_agent
4. end

Decision Rules:

1. Greetings:
   - If the user greets (hi, hey, hello, good morning, good afternoon, etc.):
     → Respond: “Hi there! How can I help you today?”
     → Route directly to `final_output_agent`. 

2. Content generation:
   - If the query asks for or mentions creating any text content:
     posts, captions, blogs, articles, ads, marketing copy, taglines, or text content:
     → Route to `validation_agent`.

3. Image requests:
   - If the query includes any image related words:
     creating images, banners, logos, designs, posters, thumbnails, or visual assets:
     → Route directly to `image_agent`.

4. Mixed queries (text + image):
   - If the user asks for both text and images in the same request:
     → Route to `image_agent`.
""",
                ),
                (
                    "human",
                    """query: {query}""",
                ),
            ]
        )

    async def decide(self, query: str) -> str:
        prompt_vars = {
            "query": query,
        }
        result = await self.llm.generate(
            prompt=self.prompt,
            input=prompt_vars,
            output_schema=SupervisorDecision,
        )
        return result["next_agent"]
