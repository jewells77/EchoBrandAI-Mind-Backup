from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure

from app.core.logger import logger
from app.config import settings

# Global database client
client: AsyncIOMotorClient = None


async def connect_to_mongo():
    """
    Creates a connection to MongoDB using the settings from config.
    Sets up the global client variable.
    """
    global client
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URI)

        # Verify connection is working by issuing a command
        await client.admin.command("ping")
        logger.info(f"Connected to MongoDB at {settings.MONGODB_URI}")
    except ConnectionFailure as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise


def get_database():
    """
    Returns the database instance.
    """
    if client is None:
        raise ConnectionError("MongoDB client not initialized")
    return client[settings.MONGODB_DB_NAME]


async def close_mongo_connection():
    """
    Closes the MongoDB connection.
    """
    global client
    if client:
        client.close()
        client = None
        logger.info("MongoDB connection closed")
