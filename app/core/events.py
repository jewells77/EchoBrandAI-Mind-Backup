from fastapi import FastAPI
from app.core.logger import logger
from app.infrastructure.db import mongodb
from app.infrastructure.vectorstores import qdrant_config, qdrant_store


async def startup_event_handler(fastapi_app: FastAPI):
    logger.info("Starting up the application...")

    try:
        # MongoDB
        fastapi_app.state.mongo_client = await mongodb.connect_to_mongo()
        logger.info("Connected to MongoDB")

        # Qdrant
        fastapi_app.state.qdrant_client = qdrant_config.init_qdrant()
        qdrant_store.ensure_all_collections_exist()
        logger.info("Qdrant initialized successfully")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Ensure we close MongoDB if startup fails midway
        await mongodb.close_mongo_connection(fastapi_app)
        raise


async def shutdown_event_handler(fastapi_app: FastAPI):
    logger.info("Shutting down the application...")

    await mongodb.close_mongo_connection(fastapi_app)
    logger.info("Disconnected from MongoDB")
