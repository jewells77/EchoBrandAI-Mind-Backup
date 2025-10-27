import asyncio
import uuid
from typing import (
    Dict,
    Any,
    List,
    TypedDict,
    Annotated,
    Union,
    Literal,
)

from typing_extensions import NotRequired


from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


from app.config import settings
from app.domain.tools.url_extractors import extract_query_and_azure_media_links
from app.infrastructure.db.langgraph_memory import LangGraphMemoryHandler
from app.infrastructure.vectorstores.qdrant_store import QdrantStore

from app.domain.agents.supervisor_agent import SupervisorAgent
from app.domain.agents.validation_agent import ValidationAgent
from app.domain.agents.image_generation_agent import ImageGenerationAgent
from app.domain.agents.competitor_analyzer import CompetitorIntelligenceAgent
from app.domain.agents.brand_analyzer import BrandAnalysisAgent
from app.domain.agents.final_output import FinalOutputAgent
from app.domain.llm_providers.base import BaseLLMProvider

from app.api.exceptions import APIError
from app.domain.llm_providers.embedding_factory import (
    get_embedding_provider,
    get_embedding_config,
)

from app.domain.tools.qdrant_helpers import extract_qdrant_texts
from app.core.logger import logger
from app.domain.utils.chat_utils import get_last_n_chats

# Node name constants for content workflow
SUPERVISOR_NODE = "supervisor"
IMAGE_AGENT_NODE = "image_agent"
VALIDATION_AGENT_NODE = "validation_agent"
EXTRACTED_DATA_NODE = "extracted_data"
EXTRACTED_COMPETITOR_NODE = "extracted_competitor_agent"
EXTRACTED_BRAND_DETAIL_NODE = "extracted_brand_detail_agent"
FIND_TRENDS_NODE = "find_trends"
FINAL_OUTPUT_NODE = "final_output_agent"
END_NODE = "end"


# import time
# import functools
# def timeit_node(func):
#     @functools.wraps(func)
#     async def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = await func(*args, **kwargs)  # await if func is async
#         end_time = time.time()
#         print(f"Time taken {func.__name__}: took {end_time - start_time:.2f} seconds")
#         return result

#     return wrapper

# Type definitions for the state
# In parallel nodes, return only updated keys; immutable/single-value keys must not be overwritten,
# or use Annotated[..., add] to safely merge multiple parallel values.


class WorkflowState(TypedDict):
    """State of the content creation workflow."""

    messages: Annotated[List[AIMessage | HumanMessage], add_messages]
    ai_generated_images: NotRequired[List[str]]
    # uploaded_images: NotRequired[List[str]]

    # Inputs
    brand_details: str
    user_qurey: str
    user_id: str

    # Process state
    step: Literal[
        SUPERVISOR_NODE,
        IMAGE_AGENT_NODE,
        VALIDATION_AGENT_NODE,
        # EXTRACTED_DATA_NODE,
        EXTRACTED_COMPETITOR_NODE,
        EXTRACTED_BRAND_DETAIL_NODE,
        FIND_TRENDS_NODE,
        FINAL_OUTPUT_NODE,
        END_NODE,
    ]
    status: Literal["running", "completed", "error"]

    # Outputs from each step
    trend_summary: NotRequired[str]
    brand_summary: NotRequired[str]
    competitor_insights: NotRequired[str]
    content_strategy: NotRequired[Dict[str, Any]]
    content_draft: NotRequired[Dict[str, Any]]
    final_content: NotRequired[Dict[str, Any]]
    final_output: str

    # Error handling
    error: NotRequired[str]
    # next_agent can be string or list of strings. next_agent -> ["node_b", "node_c"] will be parallel execution of node_b and node_c.
    next_agent: NotRequired[Union[str, List[str]]]
    is_validate: NotRequired[bool] = False


class LangGraphContentWorkflow:
    """
    Content creation workflow using LangGraph for advanced orchestration.
    This enables parallelization of tasks and complex branching logic.
    """

    def __init__(self, llm: BaseLLMProvider, scraper=None):
        self.llm = llm
        self.qdrant_store = QdrantStore()

        # Initialize agents
        self.supervisor_agent = SupervisorAgent(llm)
        self.validation_agent = ValidationAgent(llm)
        self.image_agent = ImageGenerationAgent()
        self.brand_agent = BrandAnalysisAgent(llm)
        self.final_output_agent = FinalOutputAgent(llm)
        self.competitor_agent = CompetitorIntelligenceAgent(llm)
        # Build the graph builder
        self.graph_builder = self._build_graph()

    def _build_graph(self):
        """Agentic cyclic workflow with supervisor as both policy and router."""
        builder = StateGraph(WorkflowState)

        # Define nodes
        builder.add_node(SUPERVISOR_NODE, self._supervisor_node)
        builder.add_node(IMAGE_AGENT_NODE, self._image_node)
        builder.add_node(VALIDATION_AGENT_NODE, self._validation_node)
        builder.add_node(EXTRACTED_COMPETITOR_NODE, self._extracted_competitor_node)
        builder.add_node(EXTRACTED_BRAND_DETAIL_NODE, self._extracted_brand_detail_node)
        # builder.add_node(EXTRACTED_DATA_NODE, self._extracted_data_node)
        builder.add_node(FIND_TRENDS_NODE, self._find_trends)
        builder.add_node(FINAL_OUTPUT_NODE, self._final_output_node)
        # Supervisor node decides next agent and stores in state
        builder.add_conditional_edges(
            SUPERVISOR_NODE,
            self._supervisor_router,
            {
                "validation_agent": VALIDATION_AGENT_NODE,
                "image_agent": IMAGE_AGENT_NODE,
                "final_output_agent": FINAL_OUTPUT_NODE,
            },
        )
        # Parellel edges after validation agent node (Fan out edges)
        builder.add_edge(VALIDATION_AGENT_NODE, EXTRACTED_COMPETITOR_NODE)
        builder.add_edge(VALIDATION_AGENT_NODE, EXTRACTED_BRAND_DETAIL_NODE)
        builder.add_edge(VALIDATION_AGENT_NODE, FIND_TRENDS_NODE)

        builder.add_edge(EXTRACTED_COMPETITOR_NODE, FINAL_OUTPUT_NODE)
        builder.add_edge(EXTRACTED_BRAND_DETAIL_NODE, FINAL_OUTPUT_NODE)
        builder.add_edge(FIND_TRENDS_NODE, FINAL_OUTPUT_NODE)
        # builder.add_conditional_edges(
        #     FIND_TRENDS_NODE,
        #     lambda state: state.get(
        #         "next_agent", END_NODE
        #     ),  # Short function to return the next agent based on the state
        # )

        # Image and final nodes end workflow
        builder.add_edge(IMAGE_AGENT_NODE, END)
        builder.add_edge(FINAL_OUTPUT_NODE, END)

        # Set entry point
        builder.set_entry_point(SUPERVISOR_NODE)

        return builder

    # --------------------
    # Agent Nodes (stubs):

    # Supervisor router reads next_agent from state
    def _supervisor_router(self, state: WorkflowState) -> str:
        return state.get("next_agent", END_NODE)

    async def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Supervisor decides next agent and stores in state."""
        query = state.get("user_qurey", "")
        next_agent = await self.supervisor_agent.decide(query)
        return {**state, "next_agent": next_agent, "step": SUPERVISOR_NODE}

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
            "step": IMAGE_AGENT_NODE,
            "status": "completed",
            "next_agent": END_NODE,
        }

    async def _validation_node(self, state: WorkflowState) -> WorkflowState:
        """Validation node decides next agent based on state."""
        is_validate = state.get("is_validate")
        user_id = state.get("user_id")
        website_content_collection_name = settings.QDRANT_WEBSITE_CONTENT_COLLECTION
        brand_detail_collection_name = settings.QDRANT_BRAND_DETAIL_COLLECTION
        if not is_validate:
            # Run data existence checks in parallel
            results = await asyncio.gather(
                self.qdrant_store.has_data_for_user(
                    website_content_collection_name, user_id
                ),
                self.qdrant_store.has_data_for_user(
                    brand_detail_collection_name, user_id
                ),
                return_exceptions=True,
            )
            collection_names = [
                website_content_collection_name,
                brand_detail_collection_name,
            ]
            flags = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"[ValidationNode] has_data_for_user failed for {collection_names[idx]} "
                        f"(user_id={user_id}): {result!r}"
                    )
                    flags.append(False)
                else:
                    flags.append(result)
            is_competitor_site_data_exist, is_brand_detail_data_exist = flags

            user_query = state.get("user_qurey")
            validation_result = await self.validation_agent.decide(
                is_brand_detail=is_brand_detail_data_exist,
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
                    "step": VALIDATION_AGENT_NODE,
                    "next_agent": END_NODE,
                }

        return {
            **state,
            "is_validate": is_validate,
            "step": VALIDATION_AGENT_NODE,
            "status": "completed",
            "next_agent": [
                EXTRACTED_COMPETITOR_NODE,
                EXTRACTED_BRAND_DETAIL_NODE,
                FIND_TRENDS_NODE,
            ],
        }

    async def _extracted_competitor_node(self, state: WorkflowState) -> WorkflowState:
        """Extracted competitor data."""
        user_id = state.get("user_id")
        user_query = state.get("user_qurey")
        website_content_collection_name = settings.QDRANT_WEBSITE_CONTENT_COLLECTION

        embedding_provider_name = get_embedding_config()[0]
        embedding_provider = get_embedding_provider(embedding_provider_name)
        query_vector = embedding_provider.embed_query(user_query)
        filter_dict = {"must": [{"key": "user_id", "match": {"value": user_id}}]}
        top = 10

        # Fetch website and brand detail vectors in parallel with robust error logging
        results = await self.qdrant_store.query_points_by_filter(
            website_content_collection_name,
            vector=query_vector,
            top=top,
            filter_dict=filter_dict,
        )
        limit = 3
        extracted_website_content_texts = extract_qdrant_texts(
            results=results, limit=limit
        )
        competitor_insights = await self.competitor_agent.analyze(
            competitors_data=extracted_website_content_texts
        )
        return {
            "competitor_insights": competitor_insights,
        }

    async def _extracted_brand_detail_node(self, state: WorkflowState) -> WorkflowState:
        """Extracted brand detail data."""
        user_id = state.get("user_id")
        user_query = state.get("user_qurey")
        brand_detail_collection_name = settings.QDRANT_BRAND_DETAIL_COLLECTION

        embedding_provider_name = get_embedding_config()[0]
        embedding_provider = get_embedding_provider(embedding_provider_name)
        query_vector = embedding_provider.embed_query(user_query)
        filter_dict = {"must": [{"key": "user_id", "match": {"value": user_id}}]}
        top = 10

        # Fetch website and brand detail vectors in parallel with robust error logging
        results = await self.qdrant_store.query_points_by_filter(
            brand_detail_collection_name,
            vector=query_vector,
            top=top,
            filter_dict=filter_dict,
        )

        limit = 3
        extracted_brand_detail_texts = extract_qdrant_texts(
            results=results, limit=limit
        )
        brand_summary = await self.brand_agent.analyze(
            brand_data=extracted_brand_detail_texts
        )
        return {
            "brand_summary": brand_summary,
        }

    async def _find_trends(self, state: WorkflowState) -> WorkflowState:
        """Find trends node."""
        user_query = state.get("user_qurey")
        trend_summary = "Trends found"
        return {
            "trend_summary": trend_summary,
        }

    async def _final_output_node(self, state: WorkflowState) -> WorkflowState:
        """Final output node."""
        user_message = state.get("user_qurey", "")
        trend_summary = state.get("trend_summary", "")
        brand_profile = state.get("brand_summary", "")
        competitor_insights = state.get("competitor_insights", "")
        messages = state.get("messages", [])
        last_messages = get_last_n_chats(messages, n=15)
        if last_messages:
            last_messages.pop()  # Remove last user message for context. it will be same as "user_message"

        llm_response = await self.final_output_agent.respond(
            user_message=user_message,
            brand_profile=brand_profile,
            competitor_insights=competitor_insights,
            messages=last_messages,
        )
        return {
            **state,
            "final_output": llm_response or "",
            "messages": [AIMessage(content=llm_response)],
            "step": FINAL_OUTPUT_NODE,
            "status": "completed",
            "next_agent": END_NODE,
        }

    async def run(
        self,
        brand_details: str,
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
