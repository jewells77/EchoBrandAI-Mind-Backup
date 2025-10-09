from app.core.logger import logger
from app.infrastructure.db.mongodb import connect_to_mongo, close_mongo_connection
from app.infrastructure.vectorstores import qdrant_store
from app.config import settings


async def startup_event_handler():
    """
    Function to handle startup events.
    - Initialize database connections
    - Set up any necessary resources
    """
    logger.info("Starting up the application...")

    # Connect to MongoDB
    await connect_to_mongo()
    logger.info("Connected to MongoDB")

    # Ensure all Qdrant collections exist
    for collection in settings.QDRANT_COLLECTIONS:
        collection_name = collection["name"]
        if not qdrant_store.collection_exists(collection_name):
            qdrant_store.create_collection(collection_name)
            qdrant_store.create_payload_index(collection_name, "url")
            qdrant_store.create_payload_index(collection_name, "user_id")
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Qdrant collection already exists: {collection_name}")


async def shutdown_event_handler():
    """
    Function to handle shutdown events.
    - Close database connections
    - Clean up any resources
    """
    logger.info("Shutting down the application...")

    # Close MongoDB connection
    await close_mongo_connection()
    logger.info("Disconnected from MongoDB")
