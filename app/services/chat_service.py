from typing import Dict, Any, Optional, List
import uuid

from app.domain.graphs.content_workflow import LangGraphContentWorkflow
from app.domain.llm_providers.factory import create_llm_provider
from app.infrastructure.db.langgraph_memory import LangGraphMemoryHandler


class ChatService:
    """Service for handling chat-based conversations using LangGraph workflows."""

    async def start_chat(
        self,
        brand_details: Dict[str, Any],
        user_qurey: str = "",
        competitors_summary: Optional[Dict[str, Any]] = None,
        guidelines: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Start a new chat conversation with the content generation workflow.

        Args:
            brand_details: Details about the brand
            user_qurey: Initial user query or brief (optional)
            competitors_summary: Optional pre-analyzed competitor data
            guidelines: Optional content guidelines

        Returns:
            Dict containing the workflow results and thread_id for continuation
        """
        # Create LLM provider
        llm = create_llm_provider()

        # Initialize content workflow
        workflow = LangGraphContentWorkflow(llm)

        # Generate a new thread ID for this conversation
        thread_id = f"chat_{str(uuid.uuid4())}"

        # Run the initial workflow
        result = await workflow.run(
            brand_details=brand_details,
            user_qurey=user_qurey,
            competitors_summary=competitors_summary,
            guidelines=guidelines,
            thread_id=thread_id,
        )

        return result

    async def continue_chat(
        self,
        thread_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Continue an existing chat conversation.

        Args:
            thread_id: Thread ID from the initial chat
            message: New message to continue the conversation

        Returns:
            Dict containing the response (conversational or workflow-based)
        """
        # Create LLM provider
        llm = create_llm_provider()
        return await self._handle_workflow_chat(thread_id, message, llm)

    async def _handle_workflow_chat(
        self,
        thread_id: str,
        message: str,
        llm,
    ) -> Dict[str, Any]:
        """Handle workflow-based chat interactions (original functionality)."""
        try:
            # Initialize content workflow
            workflow = LangGraphContentWorkflow(llm)

            # Get the MongoDB memory saver
            memory_saver = LangGraphMemoryHandler.get_mongodb_memory(
                thread_id=thread_id, namespace="default"
            )

            # Get the configuration for this thread
            config = LangGraphMemoryHandler.get_config(
                thread_id=thread_id,
                namespace="default",
            )

            # Compile the workflow with the MongoDB checkpointer
            compiled_workflow = workflow.graph_builder.compile(
                checkpointer=memory_saver
            )

            # Get the current state from the checkpointer
            current_state = await compiled_workflow.aget_state(config)

            if not current_state or not current_state.values:
                return {
                    "error": "No previous conversation found for this thread_id",
                    "thread_id": thread_id,
                    "status": "error",
                }

            # Create a new state with the user's message
            updated_state = {
                **current_state.values,
                "user_qurey": message,
                "step": "generation",
                "status": "running",
            }

            # Run the workflow with the updated state
            result = await compiled_workflow.ainvoke(
                updated_state,
                config=config,
            )
            # Add thread_id to the result for continuity
            result["thread_id"] = thread_id
            result["status"] = "completed"

            return result

        except Exception as e:
            return {
                "error": f"Failed to continue conversation: {str(e)}",
                "thread_id": thread_id,
                "status": "error",
            }
