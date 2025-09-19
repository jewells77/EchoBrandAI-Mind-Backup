import time
from pinecone import Pinecone, ServerlessSpec
from app.config import settings

PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_ENVIRONMENT = settings.PINECONE_ENVIRONMENT
PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine", spec=spec)
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)
