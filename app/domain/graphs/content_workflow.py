import asyncio
import uuid
from typing import (
    Dict,
    Any,
    List,
    TypedDict,
    Annotated,
    Sequence,
    Union,
    Literal,
    cast,
    get_args,
)
from typing_extensions import NotRequired

from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.domain.tools.url_extractors import extract_query_and_azure_media_links
from app.infrastructure.db.langgraph_memory import LangGraphMemoryHandler

from app.domain.agents.supervisor_agent import SupervisorAgent
from app.domain.agents.image_generation_agent import ImageGenerationAgent
from app.domain.agents.brand_dna_analyzer import BrandDNAAnalyzerAgent
from app.domain.agents.content_strategist import ContentStrategistAgent
from app.domain.agents.content_generator import ContentGeneratorAgent
from app.domain.agents.content_refiner import ContentRefinerAgent
from app.domain.agents.final_output import FinalOutputAgent
from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper
from app.api.v1.schemas.common import get_last_n_chats
from app.api.exceptions import APIError
from app.config import settings

import re


# TODO: REMOVE THIS FUNCTION AFTER TESTING
def extract_query_and_gdrive_links(text: str) -> tuple[str, list[str]]:
    """
    Extract Google Drive links from the text, convert them to direct download links,
    and return the cleaned query + list of image URLs.
    """
    # Pattern to find all drive links
    pattern = r"https?://drive\.google\.com/[^\s]+"
    drive_links = re.findall(pattern, text)

    # Convert each to direct download URL
    image_urls = []
    for link in drive_links:
        image_urls.append(gdrive_to_download_url(link))

    # Clean query (remove the links)
    cleaned_query = re.sub(pattern, "", text).strip()

    return cleaned_query, image_urls


def gdrive_to_download_url(url: str) -> str:
    """
    Convert a Google Drive share/view link to a direct download URL.
    """
    patterns = [
        r"file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"thumbnail\?id=([a-zA-Z0-9_-]+)",
    ]

    file_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            break

    if not file_id:
        raise ValueError(f"Could not extract file ID from URL: {url}")

    return f"https://drive.google.com/uc?export=download&id={file_id}"


# Type definitions for the state
class WorkflowState(TypedDict):
    """State of the content creation workflow."""

    messages: Annotated[List[AIMessage | HumanMessage], add_messages]
    # next_agent: str = "" # to be used for the supervisor agent when we have more agents
    ai_generated_images: NotRequired[List[str]]
    # uploaded_images: NotRequired[List[str]]

    # Inputs
    brand_details: Dict[str, Any]
    competitors_summary: NotRequired[Dict[str, Any]]  # Pre-analyzed competitor data
    user_qurey: str
    guidelines: Dict[str, Any]  # Can be empty

    # Process state
    step: Literal[
        "supervisor",
        "image_generation",
        "brand_analysis",
        "strategy",
        "generation",
        "refinement",
        "end",
    ]
    status: Literal["running", "completed", "error"]

    # Outputs from each step
    brand_profile: NotRequired[Dict[str, Any]]
    competitor_insights: NotRequired[Dict[str, Any]]
    content_strategy: NotRequired[Dict[str, Any]]
    content_draft: NotRequired[Dict[str, Any]]
    final_content: NotRequired[Dict[str, Any]]
    final_output: str

    # Error handling
    error: NotRequired[str]


class LangGraphContentWorkflow:
    """
    Content creation workflow using LangGraph for advanced orchestration.
    This enables parallelization of tasks and complex branching logic.
    """

    def __init__(self, llm: BaseLLMProvider, scraper=None):
        self.llm = llm
        self.scraper = scraper or PlaywrightScraper()

        # Initialize agents
        self.supervisor_agent = SupervisorAgent(llm)
        self.image_agent = ImageGenerationAgent()
        self.brand_agent = BrandDNAAnalyzerAgent(llm)
        self.strategist_agent = ContentStrategistAgent(llm)
        self.generator_agent = ContentGeneratorAgent(llm)
        self.refiner_agent = ContentRefinerAgent(llm)
        self.final_output_agent = FinalOutputAgent(llm)

        # Build the graph builder
        self.graph_builder = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create the graph
        builder = StateGraph(WorkflowState)

        # Define the nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("image_generation", self._image_generate)
        builder.add_node("is_state_existing", self._is_state_existing)
        builder.add_node("brand_analysis", self._analyze_brand)
        builder.add_node("strategy", self._create_strategy)
        builder.add_node("generation", self._generate_content)
        builder.add_node("refinement", self._refine_content)
        builder.add_node("finalize", self._finalize_output)

        # Define the edges (workflow steps)
        # Step 0: Decide route based on whether prior state-like data exists
        builder.add_conditional_edges(
            "supervisor",
            self._supervisor_router,
            {"image_generation": "image_generation", "finalize": "finalize"},
        )

        builder.add_conditional_edges(
            "is_state_existing",
            self._state_router,
            {
                "supervisor": "supervisor",
                "brand_analysis": "brand_analysis",
            },
        )

        builder.add_edge("brand_analysis", "strategy")

        builder.add_conditional_edges(
            "strategy", self._strategy_router, {"generation": "generation", "end": END}
        )

        builder.add_edge("generation", "refinement")

        builder.add_edge("refinement", "finalize")
        builder.add_edge("finalize", END)
        builder.add_edge("image_generation", END)

        # Set is_state_existing as the entry point to choose the path
        builder.set_entry_point("is_state_existing")

        # The graph will be compiled with the checkpointer during runtime
        # Return the builder instead of the compiled graph
        return builder

    # Node implementations

    async def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Supervisor node: can update state or just mark step."""
        return {**state, "step": "supervisor"}

    async def _image_generate(self, state: WorkflowState) -> WorkflowState:
        """Image generation node."""

        user_input = state["user_qurey"]
        query, image_urls = extract_query_and_azure_media_links(user_input)
        if len(image_urls) > 3:
            raise APIError(
                "A maximum of 3 image URLs are allowed.",
                status_code=400,
            )
        llm_response = await self.image_agent.generate_image(
            prompt=query, image_urls=image_urls
        )
        final_output_text = llm_response.get("text", "")
        ai_generated_images = llm_response.get("images", [])
        return {
            **state,
            "final_output": final_output_text,
            "ai_generated_images": ai_generated_images,
            "messages": [AIMessage(content=final_output_text)],
            "step": "end",
            "status": "completed",
        }

    async def _supervisor_router(self, state: WorkflowState) -> str:
        next_agent = await self.supervisor_agent.decide(state["user_qurey"])
        return "image_generation" if next_agent == "image_agent" else "finalize"

    async def _analyze_brand(self, state: WorkflowState) -> WorkflowState:
        """Analyze the brand DNA."""
        try:
            brand_profile = await self.brand_agent.analyze(
                state["brand_details"],
            )
            return {
                **state,
                "brand_profile": brand_profile,
                "step": "brand_analysis",
                "status": "completed",
            }
        except Exception as e:
            return {
                **state,
                "step": "brand_analysis",
                "status": "error",
                "error": f"Brand analysis failed: {str(e)}",
            }

    async def _create_strategy(self, state: WorkflowState) -> WorkflowState:
        """Create content strategy."""
        # Only require brand_profile, not competitor_insights
        if "brand_profile" not in state:
            return {
                **state,
                "step": "strategy",
                "status": "error",
                "error": "Missing required brand_profile for strategy",
            }

        try:
            strategy = await self.strategist_agent.suggest_strategy(
                state["brand_profile"],
                state.get("competitor_insights", {}),
                state["user_qurey"],
            )
            return {
                **state,
                "content_strategy": strategy,
                "step": "strategy",
                "status": "completed",
            }
        except Exception as e:
            return {
                **state,
                "step": "strategy",
                "status": "error",
                "error": f"Strategy creation failed: {str(e)}",
            }

    def _strategy_router(self, state: WorkflowState) -> str:
        """Route based on strategy results."""
        if state.get("status") == "error":
            return "end"

        strategy = state.get("content_strategy", {})
        # Proceed if we have at least one title
        titles = strategy.get("titles", [])

        if titles:
            return "generation"
        else:
            # End if we don't have necessary strategy outputs
            return "end"

    def _state_router(self, state: WorkflowState) -> str:
        """Route to conversational flow if prior state-like data exists, else start workflow."""
        # Heuristic: if any downstream outputs are present, treat as existing state
        has_prior_state = any(
            key in state and bool(state.get(key))
            for key in (
                "final_output",
                "final_content",
                "content_draft",
                "content_strategy",
                "competitor_insights",
                "brand_profile",
            )
        )
        # return "finalize" if has_prior_state else "brand_analysis"
        return "supervisor"

    async def _is_state_existing(self, state: WorkflowState) -> WorkflowState:
        """No-op node used before routing; returns state unchanged."""
        return state

    async def _generate_content(self, state: WorkflowState) -> WorkflowState:
        """Generate draft content."""
        try:
            strategy = state["content_strategy"]
            brand_profile = state["brand_profile"]

            # Use the first title
            theme = strategy.get("titles", [""])[0]

            # Add brand tone and target audience
            brand_tone = brand_profile.get("brand_tone", "")
            target_audience = brand_profile.get("target_audience", "")

            # Use the user_qurey to extract word count limits
            draft = await self.generator_agent.generate_content(
                theme,
                user_qurey=state["user_qurey"],
                brand_tone=brand_tone,
                target_audience=target_audience,
            )

            return {
                **state,
                "content_draft": draft,
                "step": "generation",
                "status": "completed",
            }
        except Exception as e:
            return {
                **state,
                "step": "generation",
                "status": "error",
                "error": f"Content generation failed: {str(e)}",
            }

    async def _refine_content(self, state: WorkflowState) -> WorkflowState:
        """Refine the content."""
        try:
            draft = state["content_draft"]
            brand_profile = state["brand_profile"]
            guidelines = state["guidelines"]

            # Enhance guidelines with brand profile info
            enhanced_guidelines = {**guidelines}
            if "tone" not in enhanced_guidelines:
                enhanced_guidelines["tone"] = brand_profile.get("brand_tone", "")
            if "target_audience" not in enhanced_guidelines:
                enhanced_guidelines["target_audience"] = brand_profile.get(
                    "target_audience", ""
                )
            if "keywords" not in enhanced_guidelines and "keywords" in brand_profile:
                enhanced_guidelines["keywords"] = brand_profile.get("keywords", [])

            # Pass the user_qurey to extract word count limits
            final_content = await self.refiner_agent.refine_content(
                draft["draft"],
                enhanced_guidelines,
                user_qurey=state["user_qurey"],
            )

            return {
                **state,
                "final_content": final_content,
                "step": "refinement",
                "status": "completed",
            }
        except Exception as e:
            return {
                **state,
                "step": "refinement",
                "status": "error",
                "error": f"Content refinement failed: {str(e)}",
            }

    async def _finalize_output(self, state: WorkflowState) -> WorkflowState:
        """finalize output agent."""
        user_message = state.get("user_qurey", {})
        brand_profile = state.get("brand_profile", {})
        competitor_insights = state.get("competitor_insights", {})
        guidelines = state.get("guidelines", {})
        messages = state.get("messages", [])
        last_messages = get_last_n_chats(messages, n=15)

        # Remove last user message to keep only conversation history
        last_messages.pop()
        llm_response = await self.final_output_agent.respond(
            user_message,
            brand_profile,
            competitor_insights,
            guidelines,
            last_messages,
        )
        return {
            **state,
            "final_output": llm_response,
            "messages": [AIMessage(content=llm_response)],
            "step": "end",
            "status": "completed",
        }

    async def run(
        self,
        brand_details: Dict[str, Any],
        user_qurey: str,
        competitors_summary: Dict[str, Any] = None,
        guidelines: Dict[str, Any] = None,
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run the LangGraph workflow.

        Args:
            brand_details: Details about the brand
            user_qurey: User's content request
            competitors_summary: Optional pre-analyzed competitor data (default: None)
            guidelines: Optional content guidelines (default: None)
            thread_id: Optional thread ID for continuity (default: None)

        Returns:
            Dict containing all results from the workflow
        """
        # Set defaults for optional parameters
        guidelines = guidelines or {}
        # Initialize the state
        initial_state: WorkflowState = {
            "brand_details": brand_details,
            "user_qurey": user_qurey,
            "guidelines": guidelines,
            "messages": [HumanMessage(content=user_qurey)],
            "step": "brand_analysis",
            "status": "running",
        }

        # Add competitors_summary if provided
        if competitors_summary:
            initial_state["competitors_summary"] = competitors_summary

        # Generate a thread ID if not provided
        thread_id = thread_id or f"content_{str(uuid.uuid4())}"

        # Use MongoDB for memory persistence
        memory_saver = LangGraphMemoryHandler.get_mongodb_memory(
            thread_id=thread_id, namespace="default"
        )

        # Configure the workflow with MongoDB checkpointer
        config = LangGraphMemoryHandler.get_config(
            thread_id=thread_id,
            namespace="default",
        )

        # Compile the workflow with the MongoDB checkpointer
        workflow = self.graph_builder.compile(checkpointer=memory_saver)

        # Export visualization if requested
        if settings.SHOW_WORKFLOW_GRAPH:
            from app.domain.tools.graph_visualizer import save_workflow_graph_png

            save_workflow_graph_png(workflow)

        # Run the workflow
        result = await workflow.ainvoke(
            initial_state,
            config=config,
        )
        # Add thread_id to the result for continuity
        result["thread_id"] = thread_id
        result["message"] = user_qurey
        return result
