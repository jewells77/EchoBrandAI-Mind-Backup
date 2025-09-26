from typing import Dict, Any
from app.domain.llm_providers.factory import create_llm_provider
from app.domain.agents.finalized_post_extractor import FinalizedPostExtractor
from app.domain.graphs.content_workflow import LangGraphContentWorkflow
from app.infrastructure.db.langgraph_memory import LangGraphMemoryHandler
from app.api.v1.schemas.common import get_last_n_chats


class FinalizedPostService:
    def __init__(self):
        llm = create_llm_provider()
        self.extractor = FinalizedPostExtractor(llm)
        self.llm = llm

    async def get_finalized_post(self, thread_id: str) -> Dict[str, Any]:
        # Load workflow state from MongoDB
        workflow = LangGraphContentWorkflow(self.llm)
        memory_saver = LangGraphMemoryHandler.get_mongodb_memory(thread_id=thread_id)
        config = LangGraphMemoryHandler.get_config(thread_id=thread_id)
        compiled_workflow = workflow.graph_builder.compile(checkpointer=memory_saver)
        current_state = await compiled_workflow.aget_state(config)
        if (
            not current_state
            or not current_state.values
            or "messages" not in current_state.values
        ):
            raise ValueError(
                f"No conversation history found for thread_id: {thread_id}"
            )
        messages = current_state.values["messages"]
        conversation_history = get_last_n_chats(messages, n=15)
        return await self.extractor.finalize_post(conversation_history)
