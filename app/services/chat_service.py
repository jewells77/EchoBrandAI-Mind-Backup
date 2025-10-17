from typing import Dict, Any, Optional, List
import uuid

from app.api.exceptions import APIError
from app.api.v1.schemas.common import compare_message_role_count
from app.domain.graphs.content_workflow import LangGraphContentWorkflow
from app.domain.llm_providers.factory import create_llm_provider
from app.infrastructure.db.langgraph_memory import LangGraphMemoryHandler
from langchain.schema import HumanMessage


class ChatService:
    """Service for handling chat-based conversations using LangGraph workflows."""

    async def start_chat(
        self,
        brand_details: str,
        user_qurey: str = "",
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Start a new chat conversation with the content generation workflow.

        Args:
            brand_details: Details about the brand
            user_qurey: Initial user query or brief (optional)

        Returns:
            Dict containing the workflow results and thread_id for continuation
        """
        # Create LLM provider
        llm = create_llm_provider()

        # Initialize content workflow
        workflow = LangGraphContentWorkflow(llm)

        # Generate a new thread ID for this conversation
        thread_id = f"chat_{str(uuid.uuid4())}"

        if not user_id:
            raise APIError("User ID is required for validation.", status_code=400)
        # Run the initial workflow
        result = await workflow.run(
            brand_details=brand_details,
            user_qurey=user_qurey,
            user_id=user_id,
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
            all_messages = current_state.values["messages"]
            chat_history = compare_message_role_count(
                all_messages, role="human", op=">=", threshold=80
            )
            if chat_history:
                raise APIError(
                    "Message limit exceeded for this conversation. Please start a new chat to continue.",
                    status_code=429,  # 429 Too Many Requests
                )
            # Create a new state with the user's message
            updated_state = {
                **current_state.values,
                "messages": [HumanMessage(content=message)],
                "user_qurey": message,
                "step": "generation",
                "status": "running",
                "ai_generated_images": [],
            }

            # Run the workflow with the updated state
            result = await compiled_workflow.ainvoke(
                updated_state,
                config=config,
            )
            # Add thread_id to the result for continuity
            result["thread_id"] = thread_id
            result["message"] = updated_state.get("user_qurey", message)
            result["status"] = "completed"
            return result

        except Exception as e:
            raise APIError(
                f"Failed to continue conversation: {str(e)}", status_code=500
            )

    async def get_chat_history(self, thread_id: str) -> Dict[str, Any]:
        """
        Retrieve the conversation history (messages) for a given thread_id from MongoDB.
        """
        try:
            llm = create_llm_provider()
            workflow = LangGraphContentWorkflow(llm)
            memory_saver = LangGraphMemoryHandler.get_mongodb_memory(
                thread_id=thread_id, namespace="default"
            )
            config = LangGraphMemoryHandler.get_config(
                thread_id=thread_id,
                namespace="default",
            )
            compiled_workflow = workflow.graph_builder.compile(
                checkpointer=memory_saver
            )
            state = await compiled_workflow.aget_state(config)
            if not state or not state.values or "messages" not in state.values:
                raise APIError(
                    "No conversation found for this thread_id", status_code=404
                )
            messages = state.values["messages"]
            return {"thread_id": thread_id, "messages": messages, "status": "ok"}
        except Exception as e:
            raise APIError(
                f"Could not retrieve chat history: {str(e)}", status_code=500
            )
