from typing import Dict, Any, Optional
import uuid
import asyncio
from contextlib import asynccontextmanager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from app.config import settings


class LangGraphMemoryHandler:
    """Handler for LangGraph memory operations using MongoDB."""

    @staticmethod
    @asynccontextmanager
    async def get_mongodb_memory(
        thread_id: Optional[str] = None,
        namespace: str = "default",
    ):
        """
        Get a MongoDB memory saver for LangGraph.

        Args:
            thread_id: Unique identifier for the conversation thread.
                       If None, a random UUID will be generated.
            namespace: Namespace for the checkpoint. Defaults to 'default'.

        Yields:
            AsyncMongoDBSaver: An instance of the MongoDB memory saver.
        """
        thread_id = thread_id or f"thread_{uuid.uuid4()}"

        async with AsyncMongoDBSaver.from_conn_string(
            conn_string=settings.MONGODB_URI,
            db_name=settings.MONGODB_DB_NAME,
            checkpoint_collection_name=settings.MONGODB_CHECKPOINT_COLLECTION,
            writes_collection_name=settings.MONGODB_WRITES_COLLECTION,
        ) as saver:
            yield saver

    @staticmethod
    def get_config(
        thread_id: Optional[str] = None,
        namespace: str = "default",
        callbacks: list = None,
    ) -> Dict[str, Any]:
        """
        Create a configuration dictionary for LangGraph with MongoDB checkpointing.

        Args:
            thread_id: Unique identifier for the conversation thread.
                       If None, a random UUID will be generated.
            namespace: Namespace for the checkpoint. Defaults to 'default'.
            callbacks: List of callbacks to add to the configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary for LangGraph.
        """
        thread_id = thread_id or f"thread_{uuid.uuid4()}"

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": namespace,
            }
        }

        if callbacks:
            config["callbacks"] = callbacks

        return config
