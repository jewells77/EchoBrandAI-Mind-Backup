import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch

from app.domain.graphs.content_workflow import LangGraphContentWorkflow
from app.domain.llm_providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    async def agenerate(self, prompt, **kwargs):
        """Mock generation that returns predefined responses based on the prompt."""
        if "brand" in prompt.lower():
            return {
                "brand_tone": "professional",
                "target_audience": "eco-conscious consumers",
            }
        elif "competitor" in prompt.lower():
            return {"strengths": ["quality"], "weaknesses": ["pricing"]}
        elif "strategy" in prompt.lower():
            return {"titles": ["How to Go Green"], "formats": ["blog post"]}
        elif "generate" in prompt.lower():
            return {"draft": "This is a draft blog post about going green."}
        elif "refine" in prompt.lower():
            return {"final": "This is a refined blog post about going green."}
        return {"response": "Mock response"}


@pytest.mark.asyncio
@patch("app.domain.graphs.content_workflow.MongoDBSaver")
async def test_langgraph_workflow_with_mongodb(mock_mongodb_saver):
    """Test the LangGraph workflow with MongoDB persistence."""
    # Mock MongoDB saver
    mock_saver_instance = MagicMock()
    mock_mongodb_saver.return_value = mock_saver_instance

    # Mock environment variables for testing
    with patch.dict(
        os.environ,
        {"MONGODB_URI": "mongodb://testhost:27017", "MONGODB_DB_NAME": "test_db"},
    ):
        # Create the workflow with mock LLM
        mock_llm = MockLLMProvider()
        workflow = LangGraphContentWorkflow(mock_llm)

        # Test input data
        brand_details = {"name": "EcoTest", "values": ["sustainability"]}
        user_qurey = "Create a blog post about sustainability"

        # Run the workflow
        result = await workflow.run(
            brand_details=brand_details,
            user_qurey=user_qurey,
            competitors=["https://example.com"],
            guidelines={"keywords": ["green"]},
        )

        # Verify MongoDB saver was used
        mock_mongodb_saver.assert_called_once()

        # Verify basic structure of result
        assert "brand_profile" in result
        assert "competitor_insights" in result
        assert "content_strategy" in result
        assert "content_draft" in result
        assert "final_content" in result
        assert "thread_id" in result
