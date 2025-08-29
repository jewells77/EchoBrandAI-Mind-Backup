from typing import Dict, Any, List, Optional

from app.domain.graphs.content_workflow import LangGraphContentWorkflow
from app.domain.llm_providers.factory import create_llm_provider


class ContentService:
    """Service for content creation using LangGraph workflows."""

    async def generate_content(
        self,
        brand_details: Dict[str, Any],
        content_request: str,
        competitors: List[str] = None,
        guidelines: Dict[str, Any] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate content using the LangGraph workflow.

        Args:
            brand_details: Details about the brand
            content_request: Content brief or request
            competitors: Optional list of competitor URLs
            guidelines: Optional content guidelines
            thread_id: Optional thread ID to continue a previous conversation

        Returns:
            Dict containing the generated content and workflow metadata including thread_id
        """
        # Create LLM provider
        llm = create_llm_provider()

        # Initialize content workflow
        workflow = LangGraphContentWorkflow(llm)

        # Run the workflow with the provided thread_id (if any)
        result = await workflow.run(
            brand_details=brand_details,
            content_request=content_request,
            competitors=competitors,
            guidelines=guidelines,
            thread_id=thread_id,
        )

        # Result will contain the thread_id added by the workflow
        return result
