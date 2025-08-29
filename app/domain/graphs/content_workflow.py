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
    competitors: List[str]  # Can be empty
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
        competitors: List[str] = None,
        guidelines: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate the multi-agent workflow for content creation.

        Args:
            brand_details: Dictionary containing brand information
            content_request: Content request or brief
            competitors: Optional list of competitor URLs (default: None)
            guidelines: Optional content guidelines (default: None)

        Returns:
            Dictionary containing all outputs from the workflow
        """
        # Set defaults for optional parameters
        competitors = competitors or []
        guidelines = guidelines or {}
        # Step 1: Analyze brand DNA
        brand_profile = await self.brand_agent.analyze(brand_details, competitors)

        # Step 2: Fetch competitor insights (can be parallelized)
        competitor_insights = await self.competitor_agent.summarize_competitors(
            competitors
        )

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

        draft = await self.generator_agent.generate_content(
            theme, format_, brand_tone=brand_tone, target_audience=target_audience
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

        final_content = await self.refiner_agent.refine_content(
            draft["draft"], enhanced_guidelines
        )

        # Return a composite result with all the outputs
        return {
            "brand_profile": brand_profile,
            "competitor_insights": competitor_insights,
            "content_strategy": strategy,
            "content_draft": draft,
            "final_content": final_content,
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
            competitor_insights = await self.competitor_agent.summarize_competitors(
                state["competitors"]
            )
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

            draft = await self.generator_agent.generate_content(
                theme, format_, brand_tone=brand_tone, target_audience=target_audience
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

            final_content = await self.refiner_agent.refine_content(
                draft["draft"], enhanced_guidelines
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
        competitors: List[str] = None,
        guidelines: Dict[str, Any] = None,
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """
        Run the LangGraph workflow.

        Args:
            brand_details: Details about the brand
            content_request: User's content request
            competitors: Optional list of competitor URLs (default: None)
            guidelines: Optional content guidelines (default: None)
            thread_id: Optional thread ID for continuity (default: None)

        Returns:
            Dict containing all results from the workflow
        """
        # Set defaults for optional parameters
        competitors = competitors or []
        guidelines = guidelines or {}
        # Initialize the state
        initial_state: WorkflowState = {
            "brand_details": brand_details,
            "competitors": competitors,
            "content_request": content_request,
            "guidelines": guidelines,
            "step": "brand_analysis",
            "status": "running",
        }

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
