from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure

from app.core.logger import logger
from app.config import settings

_mongo_client: AsyncIOMotorClient | None = None


async def connect_to_mongo():
    """
    Initialize the MongoDB client for this FastAPI worker.

    How it works:
    1. Each worker runs its own startup event, so this function is called once per worker.
    2. _mongo_client is set per worker, and stored in app.state.mongo_client for helpers/routes.
    3. AsyncIOMotorClient is designed to be lightweight and async-safe per process.

    Why it works:
    - Within a single worker, both _mongo_client and app.state.mongo_client point to the same client.
    - Routes and helper functions can safely access the client without passing it explicitly.

    Why it can fail / won’t work if misused:
    - **Globals are not shared across workers.** Worker 1’s _mongo_client is invisible to Worker 2.
    - If you forget to call this function in the worker’s startup/lifespan, app.state.mongo_client remains None.
      Any route or helper that tries to use it will fail at runtime.
    - Any code assuming a single shared _mongo_client across workers will break.

    ✅ Best practice:
    - Always call connect_to_mongo() during FastAPI startup/lifespan events.
    - Access the client via app.state.mongo_client or _mongo_client inside routes or helpers.
    """
    try:
        global _mongo_client
        client = AsyncIOMotorClient(settings.MONGODB_URI)

        # Verify connection is working by issuing a command
        await client.admin.command("ping")
        logger.info(f"Connected to MongoDB")
        _mongo_client = client
        return client
    except ConnectionFailure as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise


def get_database() -> AsyncIOMotorDatabase:
    """
    Returns the database instance from app.state.
    """
    if not _mongo_client:
        raise RuntimeError("MongoDB client not initialized")
    return _mongo_client[settings.MONGODB_DB_NAME]


async def close_mongo_connection(fastapi_app: FastAPI):
    """
    Closes the MongoDB connection gracefully, both the per-worker global
    and app.state.
    """
    global _mongo_client

    # Close the global client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        logger.info("🛑 _mongo_client closed")

    # Close and remove from app.state
    client = getattr(fastapi_app.state, "mongo_client", None)
    if client:
        client.close()  # optional, same object as _mongo_client
        del fastapi_app.state.mongo_client
        logger.info("🛑 app.state.mongo_client cleared")
