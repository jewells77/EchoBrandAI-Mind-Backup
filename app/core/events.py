from app.core.logger import logger
from app.infrastructure.db.mongodb import connect_to_mongo, close_mongo_connection


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
