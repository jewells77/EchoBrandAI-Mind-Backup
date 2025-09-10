from typing import Dict, Any, List, Optional
from app.domain.llm_providers.base import BaseLLMProvider
from langchain.prompts import ChatPromptTemplate


class ConversationalAgent:
    """Agent for handling natural conversational interactions."""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        # self.content_prompt = content_prompt

    async def respond(
        self,
        user_message: str,
        brand_profile: Dict[str, Any],
        competitor_insights: Dict[str, Any],
        final_output: Dict[str, Any],
        guidelines: Dict[str, Any],
        # chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a conversational response based on user message and context.
        """
        # Build context from previous workflow outputs
        # context_info = self._build_context_summary(conversation_context)

        # # Build chat history for context
        # history_context = self._build_history_context(chat_history or [])

        # Conversational system prompt
        brand_str = flatten_brand_profile(brand_profile)
        competitor_str = flatten_competitor_insights(competitor_insights)
        guidelines_str = flatten_guidelines(guidelines)
        final_output_str = f"Title: {final_output.get('title','')}\nContent: {final_output.get('content','')}"
        system_prompt = f"""You are an AI assistant helping with content creation and brand strategy. 
=== Brand Profile ===
{brand_str}
\n=== Competitor Insights ===
{competitor_str}
\n=== Guidelines ===
{guidelines_str}
\n=== Final Output ===
{final_output_str}

Your role is to:
1. Have natural conversations about content creation, brand strategy, and marketing
2. Answer questions about previous work or analysis
3. Help refine, modify, or create new content based on requests
4. Provide insights and suggestions in a conversational manner
5. Ask clarifying questions when needed

IMPORTANT: When the user asks to modify existing content (like "make it shorter", "summarize in 50 words", "rewrite this"), refer to the "Most recent content created" section above and modify that content according to their request.

Respond in a natural, helpful, and conversational tone. If the user asks for specific content creation or modification, generate the requested content directly in your response rather than just describing what you would do.

CRITICAL: If brand details are present in the context, ALWAYS use the exact brand details (e.g., brand name, product line, trademark). Do NOT use placeholders like "[Brand Name]" or "[Product]". Brand details will always be provided in the context."""

        try:
            content_prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", user_message)]
            )

            result = await self.llm.generate(prompt=content_prompt, input={})
            # You now get a typed object back
            return result.content.strip()

        except Exception as e:
            return f"I apologize, but I encountered an error while processing your message: {str(e)}. Could you please try rephrasing your request?"


def flatten_brand_profile(profile: Dict[str, Any]) -> str:
    return "\n".join(
        [f"{k.replace('_', ' ').title()}: {v}" for k, v in profile.items()]
    )


def flatten_competitor_insights(ci: Dict[str, Any]) -> str:
    lines = []
    for competitor, details in ci.items():
        lines.append(f"{competitor}:")
        for k, v in details.items():
            lines.append(f"  {k.title()}: {v}")
    return "\n".join(lines)


def flatten_guidelines(guidelines: Dict[str, Any]) -> str:
    return "\n".join([f"{k.title()}: {v}" for k, v in guidelines.items()])
