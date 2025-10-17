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
    ]
    next_agent_reason: str


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
                    """
You are the SUPERVISOR agent — the main controller of an agentic LangGraph workflow.

Your responsibility:
- Analyze each user query.
- Decide which agent should act next.
- Only you make routing decisions — no other agent can.

=========================
AVAILABLE AGENTS
=========================
1. validation_agent → For text-based or content-generation tasks.
2. image_agent → For any visual or design generation tasks.
3. final_output_agent → For direct user responses or conversation closure.

=========================
ROUTING RULES
=========================

1. Greetings:
   - If the user greets (e.g., "hi", "hey", "hello", "good morning", "good afternoon", etc.):
     → Respond: “Hi there! How can I help you today?”
     → Route to `final_output_agent`.

2. Text or Content Generation:
   - If the query requests or discusses writing any text content such as:
     posts, captions, blogs, articles, ad copy, marketing content, taglines, or descriptions:
     → Route to `validation_agent`.

3. Image or Visual Creation:
   - If the query includes requests related to:
     images, banners, logos, posters, thumbnails, graphics, or any visual design:
     → Route to `image_agent`.

4. Mixed Requests (Text + Image):
   - If the query asks for both text and visuals in the same request:
     → Route to `image_agent` (since text is usually part of image context).

5. Default / Unclear Cases:
   - If the intent is unclear or conversational but not a greeting:
     → Route to `validation_agent` for interpretation and content handling.

=========================
NOTES
=========================
- Be decisive: always choose one clear route.
- Never execute the agent’s job yourself — only decide the next node.
- Your output must include both:
  1. The routing decision (agent name)
  2. A short rationale (why you made that choice)
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
