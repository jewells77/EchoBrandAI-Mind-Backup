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
from app.infrastructure.vectorstores.qdrant_store import QdrantStore

from app.domain.agents.supervisor_agent import SupervisorAgent
from app.domain.agents.validation_agent import ValidationAgent
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
from app.domain.agents.competitor_intelligence import CompetitorIntelligenceAgent
from app.domain.llm_providers.embedding_factory import get_embedding_provider
from app.config import settings

import re

from app.domain.tools.qdrant_helpers import extract_qdrant_texts


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
    ai_generated_images: NotRequired[List[str]]
    # uploaded_images: NotRequired[List[str]]

    # Inputs
    brand_details: Dict[str, Any]
    user_qurey: str
    user_id: str

    # Process state
    step: Literal[
        "supervisor",
        "brand_agent",
        "competitor_agent",
        "strategy_agent",
        "generator_agent",
        "refiner_agent",
        "image_agent",
        "validation_agent",
        "final_output_agent",
        "end",
    ]
    status: Literal["running", "completed", "error"]

    # Outputs from each step
    brand_profile: NotRequired[str]
    competitor_insights: NotRequired[str]
    content_strategy: NotRequired[Dict[str, Any]]
    content_draft: NotRequired[Dict[str, Any]]
    final_content: NotRequired[Dict[str, Any]]
    final_output: str

    # Error handling
    error: NotRequired[str]
    next_agent: NotRequired[
        str
    ]  # <--- Add core explicit next_agent for supervisor/agent routing


class LangGraphContentWorkflow:
    """
    Content creation workflow using LangGraph for advanced orchestration.
    This enables parallelization of tasks and complex branching logic.
    """

    def __init__(self, llm: BaseLLMProvider, scraper=None):
        self.llm = llm
        self.scraper = scraper or PlaywrightScraper()
        self.qdrant_store = QdrantStore()

        # Initialize agents
        self.supervisor_agent = SupervisorAgent(llm)
        self.validation_agent = ValidationAgent(llm)
        self.image_agent = ImageGenerationAgent()
        self.brand_agent = BrandDNAAnalyzerAgent(llm)
        self.strategist_agent = ContentStrategistAgent(llm)
        self.generator_agent = ContentGeneratorAgent(llm)
        self.refiner_agent = ContentRefinerAgent(llm)
        self.final_output_agent = FinalOutputAgent(llm)
        self.competitor_agent = CompetitorIntelligenceAgent(llm)
        # Build the graph builder
        self.graph_builder = self._build_graph()

    def _build_graph(self):
        """Agentic cyclic workflow with supervisor as both policy and router."""
        builder = StateGraph(WorkflowState)

        # Define nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("image_agent", self._image_node)
        builder.add_node("validation_agent", self._validation_node)
        builder.add_node("final_output_agent", self._final_output_node)
        # builder.add_node("competitor_agent", self._competitor_node)
        # Supervisor node decides next agent and stores in state
        builder.add_conditional_edges(
            "supervisor",
            self._supervisor_router,
            {
                "validation_agent": "validation_agent",
                "image_agent": "image_agent",
                "final_output_agent": "final_output_agent",
                "end": END,
            },
        )

        # Validation agent uses next_agent from state
        builder.add_conditional_edges(
            "validation_agent",
            lambda state: state.get("next_agent", END),
        )

        # Set entry point
        builder.set_entry_point("supervisor")

        # Image and final nodes end workflow
        builder.add_edge("image_agent", END)
        builder.add_edge("final_output_agent", END)

        return builder

    # --------------------
    # Agent Nodes (stubs):

    # Supervisor router reads next_agent from state
    def _supervisor_router(self, state: WorkflowState) -> str:
        return state.get("next_agent", END)

    async def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Supervisor decides next agent and stores in state."""
        query = state.get("user_qurey", "")
        next_agent = await self.supervisor_agent.decide(query)
        return {**state, "next_agent": next_agent, "step": "supervisor"}

    async def _image_node(self, state: WorkflowState) -> WorkflowState:
        """Image generation node."""
        user_input = state["user_qurey"]
        query, image_urls = extract_query_and_azure_media_links(user_input)
        if len(image_urls) > 3:
            raise APIError("A maximum of 3 image URLs are allowed.", status_code=400)

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
            "step": "image_agent",
            "status": "completed",
            "next_agent": "end",
        }

    async def _validation_node(self, state: WorkflowState) -> WorkflowState:
        """Validation node decides next agent based on state."""
        user_id = state.get("user_id")
        collection_name = settings.QDRANT_WEBSITE_CONTENT_COLLECTION
        is_competitor_site_data_exist = await self.qdrant_store.has_data_for_user(
            collection_name, user_id
        )
        is_brand_detail = True
        user_query = state.get("user_qurey")
        validation_result = await self.validation_agent.decide(
            is_brand_detail=is_brand_detail,
            is_competitor_site=is_competitor_site_data_exist,
            user_query=user_query,
        )
        is_validate = validation_result["is_validate"]
        message = validation_result["message"]
        if not is_validate:
            return {
                **state,
                "validated_status": is_validate,
                "final_output": message,
                "messages": [AIMessage(content=message)],
                "step": "validation_agent",
                "next_agent": "end",
            }

        user_query = state.get("user_qurey")
        embedding_provider = get_embedding_provider("huggingface")
        query_vector = embedding_provider.get_embedding(user_query)
        filter_dict = {"must": [{"key": "user_id", "match": {"value": user_id}}]}
        results = await self.qdrant_store.query_points_by_filter(
            collection_name, vector=query_vector, top=10, filter_dict=filter_dict
        )
        extracted_texts = extract_qdrant_texts(results=results, limit=3)
        competitor_insights = await self.competitor_agent.decide(extracted_texts)
        return {
            **state,
            "competitor_insights": competitor_insights,
            "step": "validation_agent",
            "next_agent": "final_output_agent",
        }

    async def _final_output_node(self, state: WorkflowState) -> WorkflowState:
        """Final output node."""
        user_message = state.get("user_qurey", "")
        brand_profile = state.get("brand_profile", "")
        competitor_insights = state.get("competitor_insights", "")
        messages = state.get("messages", [])
        last_messages = get_last_n_chats(messages, n=15)
        if last_messages:
            last_messages.pop()  # Remove last user message for context

        llm_response = await self.final_output_agent.respond(
            user_message, brand_profile, competitor_insights, last_messages
        )
        return {
            **state,
            "final_output": llm_response or "",
            "messages": [AIMessage(content=llm_response)],
            "step": "final_output_agent",
            "status": "completed",
            "next_agent": "end",
        }

    async def run(
        self,
        brand_details: Dict[str, Any],
        user_qurey: str,
        user_id: str = None,
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run the LangGraph workflow.

        Args:
            brand_details: Details about the brand
            user_qurey: User's content request
            thread_id: Optional thread ID for continuity (default: None)
            user_id: User ID for validation

        Returns:
            Dict containing all results from the workflow
        """

        # Initialize the state
        initial_state: WorkflowState = {
            "brand_details": brand_details,
            "user_qurey": user_qurey,
            "user_id": user_id,
            "messages": [HumanMessage(content=user_qurey)],
            "step": "brand_analysis",
            "status": "running",
        }

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
