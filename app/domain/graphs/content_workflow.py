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

from langchain.schema import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.infrastructure.db.langgraph_memory import LangGraphMemoryHandler

from app.domain.agents.brand_dna_analyzer import BrandDNAAnalyzerAgent
from app.domain.agents.competitor_intelligence import CompetitorIntelligenceAgent
from app.domain.agents.content_strategist import ContentStrategistAgent
from app.domain.agents.content_generator import ContentGeneratorAgent
from app.domain.agents.content_refiner import ContentRefinerAgent
from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper


# Type definitions for the state
class WorkflowState(TypedDict):
    """State of the content creation workflow."""

    # Inputs
    brand_details: Dict[str, Any]
    competitors_summary: NotRequired[Dict[str, Any]]  # Pre-analyzed competitor data
    content_request: str
    guidelines: Dict[str, Any]  # Can be empty

    # Process state
    step: Literal[
        "brand_analysis",
        "competitor_analysis",
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

    # Error handling
    error: NotRequired[str]


class ContentCreationWorkflow:
    """Sequential workflow implementation for content creation."""

    def __init__(self, llm: BaseLLMProvider, scraper=None):
        self.llm = llm
        self.scraper = scraper or PlaywrightScraper()
        self.brand_agent = BrandDNAAnalyzerAgent(llm)
        self.competitor_agent = CompetitorIntelligenceAgent(llm, scraper)
        self.strategist_agent = ContentStrategistAgent(llm)
        self.generator_agent = ContentGeneratorAgent(llm)
        self.refiner_agent = ContentRefinerAgent(llm)

    async def run(
        self,
        brand_details: Dict[str, Any],
        content_request: str,
        competitors_summary: Dict[str, Any] = None,
        guidelines: Dict[str, Any] = None,
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate the multi-agent workflow for content creation.

        Args:
            brand_details: Dictionary containing brand information
            content_request: Content request or brief
            competitors_summary: Optional pre-analyzed competitor data (default: None)
            guidelines: Optional content guidelines (default: None)
            thread_id: Optional thread ID for continuity (default: None)

        Returns:
            Dictionary containing all outputs from the workflow
        """
        # Set defaults for optional parameters
        guidelines = guidelines or {}
        # Step 1: Analyze brand DNA - pass empty list for competitors since we're using summary
        brand_profile = await self.brand_agent.analyze(brand_details, [])

        # Step 2: Use competitor insights summary
        if competitors_summary:
            competitor_insights = competitors_summary
        else:
            # If no summary provided, create empty placeholder
            competitor_insights = {
                "competitor_insights": [],
                "content_gaps": ["No competitor data provided for analysis"],
                "trending_topics": [],
                "content_types": [],
            }

        # Step 3: Suggest content strategy
        strategy = await self.strategist_agent.suggest_strategy(
            brand_profile, competitor_insights, content_request
        )

        # Step 4: Generate content draft
        # For simplicity, use the first theme/format
        theme = strategy.get("titles", [""])[0]
        format_ = strategy.get("formats", [""])[0]

        # Add brand tone and target audience from the brand profile
        brand_tone = brand_profile.get("brand_tone", "")
        target_audience = brand_profile.get("target_audience", "")

        # Pass the original content request to extract word count limits
        draft = await self.generator_agent.generate_content(
            theme,
            format_,
            content_request=content_request,
            brand_tone=brand_tone,
            target_audience=target_audience,
        )

        # Step 5: Refine content
        # Enhance guidelines with brand profile info
        enhanced_guidelines = {**guidelines}
        if "tone" not in enhanced_guidelines and brand_tone:
            enhanced_guidelines["tone"] = brand_tone
        if "target_audience" not in enhanced_guidelines and target_audience:
            enhanced_guidelines["target_audience"] = target_audience
        if "keywords" not in enhanced_guidelines and "keywords" in brand_profile:
            enhanced_guidelines["keywords"] = brand_profile.get("keywords", [])

        # Pass the content request to the refiner to extract word count limits
        final_content = await self.refiner_agent.refine_content(
            draft["draft"], enhanced_guidelines, content_request=content_request
        )

        # Generate a thread ID if not provided
        thread_id = thread_id or f"content_{str(uuid.uuid4())}"

        # Return a composite result with all the outputs
        return {
            "brand_profile": brand_profile,
            "competitor_insights": competitor_insights,
            "content_strategy": strategy,
            "content_draft": draft,
            "final_content": final_content,
            "thread_id": thread_id,
        }


class LangGraphContentWorkflow:
    """
    Content creation workflow using LangGraph for advanced orchestration.
    This enables parallelization of tasks and complex branching logic.
    """

    def __init__(self, llm: BaseLLMProvider, scraper=None):
        self.llm = llm
        self.scraper = scraper or PlaywrightScraper()

        # Initialize agents
        self.brand_agent = BrandDNAAnalyzerAgent(llm)
        self.competitor_agent = CompetitorIntelligenceAgent(llm, scraper)
        self.strategist_agent = ContentStrategistAgent(llm)
        self.generator_agent = ContentGeneratorAgent(llm)
        self.refiner_agent = ContentRefinerAgent(llm)

        # Build the graph builder
        self.graph_builder = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create the graph
        builder = StateGraph(WorkflowState)

        # Define the nodes
        builder.add_node("brand_analysis", self._analyze_brand)
        builder.add_node("competitor_analysis", self._analyze_competitors)
        builder.add_node("strategy", self._create_strategy)
        builder.add_node("generation", self._generate_content)
        builder.add_node("refinement", self._refine_content)

        # Define the edges (workflow steps)
        # Step 1: Sequential flow - brand_analysis → competitor_analysis → strategy
        builder.add_edge("brand_analysis", "competitor_analysis")
        builder.add_edge("competitor_analysis", "strategy")

        # Step 2: Strategy to generation
        builder.add_conditional_edges(
            "strategy", self._strategy_router, {"generation": "generation", "end": END}
        )

        # Step 3: Generation to refinement
        builder.add_edge("generation", "refinement")

        # Step 4: Refinement to end
        builder.add_edge("refinement", END)

        # Set brand_analysis as the entry point directly
        builder.set_entry_point("brand_analysis")

        # The graph will be compiled with the checkpointer during runtime
        # Return the builder instead of the compiled graph
        return builder

    # Node implementations
    async def _analyze_brand(self, state: WorkflowState) -> WorkflowState:
        """Analyze the brand DNA."""
        try:
            brand_profile = await self.brand_agent.analyze(
                state["brand_details"], state["competitors"]
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

    async def _analyze_competitors(self, state: WorkflowState) -> WorkflowState:
        """Analyze competitors."""
        try:
            # Use provided summary if available, otherwise use empty placeholder
            if "competitors_summary" in state and state["competitors_summary"]:
                competitor_insights = state["competitors_summary"]
            else:
                # If no summary provided, create empty placeholder
                competitor_insights = {
                    "competitor_insights": [],
                    "content_gaps": ["No competitor data provided for analysis"],
                    "trending_topics": [],
                    "content_types": [],
                }

            # Check if there was a scraping error but we have some fallback data
            if competitor_insights.get("scraping_error"):
                # Log the error but continue with the workflow using the limited data
                # This is important - we don't want to fail the entire workflow just because
                # we couldn't scrape competitors, especially if we have some data
                return {
                    **state,
                    "competitor_insights": competitor_insights,
                    "step": "competitor_analysis",
                    "status": "completed",
                    "competitor_scraping_warning": ", ".join(
                        competitor_insights.get("error_details", ["Access restricted"])
                    ),
                }

            return {
                **state,
                "competitor_insights": competitor_insights,
                "step": "competitor_analysis",
                "status": "completed",
            }
        except Exception as e:
            return {
                **state,
                "step": "competitor_analysis",
                "status": "error",
                "error": f"Competitor analysis failed: {str(e)}",
            }

    async def _create_strategy(self, state: WorkflowState) -> WorkflowState:
        """Create content strategy."""
        # Check if we have both required inputs
        if "brand_profile" not in state or "competitor_insights" not in state:
            # This shouldn't happen in a properly configured graph
            return {
                **state,
                "step": "strategy",
                "status": "error",
                "error": "Missing required inputs for strategy",
            }

        try:
            strategy = await self.strategist_agent.suggest_strategy(
                state["brand_profile"],
                state["competitor_insights"],
                state["content_request"],
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
        # Check if we have at least one title and format to proceed
        titles = strategy.get("titles", [])
        formats = strategy.get("formats", [])

        if titles and formats:
            return "generation"
        else:
            # End if we don't have necessary strategy outputs
            return "end"

    async def _generate_content(self, state: WorkflowState) -> WorkflowState:
        """Generate draft content."""
        try:
            strategy = state["content_strategy"]
            brand_profile = state["brand_profile"]

            # Use the first title and format
            theme = strategy.get("titles", [""])[0]
            format_ = strategy.get("formats", [""])[0]

            # Add brand tone and target audience
            brand_tone = brand_profile.get("brand_tone", "")
            target_audience = brand_profile.get("target_audience", "")

            # Use the content_request to extract word count limits
            draft = await self.generator_agent.generate_content(
                theme,
                format_,
                content_request=state["content_request"],
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

            # Pass the content_request to extract word count limits
            final_content = await self.refiner_agent.refine_content(
                draft["draft"],
                enhanced_guidelines,
                content_request=state["content_request"],
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

    async def run(
        self,
        brand_details: Dict[str, Any],
        content_request: str,
        competitors_summary: Dict[str, Any] = None,
        guidelines: Dict[str, Any] = None,
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run the LangGraph workflow.

        Args:
            brand_details: Details about the brand
            content_request: User's content request
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
            "competitors": [],  # Empty list as we're using competitors_summary
            "content_request": content_request,
            "guidelines": guidelines,
            "step": "brand_analysis",
            "status": "running",
        }

        # Add competitors_summary if provided
        if competitors_summary:
            initial_state["competitors_summary"] = competitors_summary

        # Generate a thread ID if not provided
        thread_id = thread_id or f"content_{str(uuid.uuid4())}"

        # Use MongoDB for memory persistence
        async with LangGraphMemoryHandler.get_mongodb_memory(
            thread_id=thread_id, namespace="content_workflow"
        ) as memory_saver:
            # Configure the workflow with MongoDB checkpointer
            config = LangGraphMemoryHandler.get_config(
                thread_id=thread_id,
                namespace="content_workflow",
            )

            # Compile the workflow with the MongoDB checkpointer
            workflow = self.graph_builder.compile(checkpointer=memory_saver)

            # Run the workflow
            result = await workflow.ainvoke(
                initial_state,
                config=config,
            )

            # Add thread_id to the result for continuity
            result["thread_id"] = thread_id
            return result
