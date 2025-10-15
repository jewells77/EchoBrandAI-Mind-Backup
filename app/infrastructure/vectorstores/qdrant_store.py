from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

from app.core.logger import logger
from app.infrastructure.vectorstores.qdrant_config import get_qdrant_client
from app.api.exceptions import APIError
from app.config import settings


class QdrantStore:
    COLLECTION_DIM = 384
    COLLECTION_DISTANCE = Distance.COSINE

    def __init__(self):
        self.client = get_qdrant_client()

    async def collection_exists(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name=collection_name)

    async def create_collection(self, collection_name: str):
        if await self.collection_exists(collection_name):
            raise APIError(
                f"Collection '{collection_name}' already exists.", status_code=400
            )
        await self.client.create_collection(
            collection_name,
            vectors_config=VectorParams(
                size=self.COLLECTION_DIM, distance=self.COLLECTION_DISTANCE
            ),
        )

    async def ensure_collection_exists(self, collection_name: str):
        if not await self.collection_exists(collection_name):
            raise APIError(
                f"Collection '{collection_name}' does not exist.", status_code=404
            )

    async def validate_payload_fields(self, collection_name: str, payload: dict):
        for c in settings.QDRANT_COLLECTIONS:
            if c["name"] == collection_name:
                required = set(c.get("required_payload", []))
                optional = set(c.get("optional_payload", []))
                allowed = required | optional
                payload_keys = set(payload.keys())

                missing = required - payload_keys
                if missing:
                    raise APIError(
                        f"Missing required payload fields for {collection_name}: {list(missing)}",
                        status_code=400,
                    )

                extra = payload_keys - allowed
                if extra:
                    raise APIError(
                        f"Unexpected payload fields for {collection_name}: {list(extra)}",
                        status_code=400,
                    )
                return
        raise APIError(
            f"Collection '{collection_name}' is not defined in QDRANT_COLLECTIONS.",
            status_code=404,
        )

    async def validate_fields_exist(self, collection_name: str, field_dict: dict):
        for c in settings.QDRANT_COLLECTIONS:
            if c["name"] == collection_name:
                allowed = set(c.get("required_payload", [])) | set(
                    c.get("optional_payload", [])
                )
                invalid = set(field_dict.keys()) - allowed
                if invalid:
                    raise APIError(
                        f"Invalid fields for {collection_name}: {list(invalid)}",
                        status_code=400,
                    )
                return
        raise APIError(
            f"Collection '{collection_name}' is not defined in QDRANT_COLLECTIONS.",
            status_code=404,
        )

    async def upsert_points(self, collection_name: str, points: list):
        return await self.client.upsert(collection_name, wait=True, points=points)

    async def query_points(
        self, collection_name: str, vector: list, top: int = 5, filter: dict = None
    ):
        return await self.client.search(
            collection_name, query_vector=vector, limit=top, query_filter=filter
        )

    async def query_points_by_filter(
        self,
        collection_name: str,
        vector: list,
        top: int = 5,
        filter_dict: dict = None,
    ):
        if filter_dict:
            field_names = set()
            for clause in ["must", "should", "must_not"]:
                for cond in filter_dict.get(clause, []):
                    field_names.add(cond["key"])
            if field_names:
                await self.validate_fields_exist(
                    collection_name, {k: None for k in field_names}
                )

            def to_field_condition(cond):
                return FieldCondition(
                    key=cond["key"], match=MatchValue(**cond["match"])
                )

            filter_kwargs = {}
            for clause in ["must", "should", "must_not"]:
                if clause in filter_dict:
                    filter_kwargs[clause] = [
                        to_field_condition(c) for c in filter_dict[clause]
                    ]
            qdrant_filter = Filter(**filter_kwargs)
        else:
            qdrant_filter = None

        return await self.client.search(
            collection_name, query_vector=vector, limit=top, query_filter=qdrant_filter
        )

    async def delete_points(self, collection_name: str, point_ids: list):
        return await self.client.delete(
            collection_name, points_selector={"points": point_ids}
        )

    async def delete_points_by_filter(self, collection_name: str, filter_dict: dict):
        field_names = set()
        for clause in ["must", "should", "must_not"]:
            for cond in filter_dict.get(clause, []):
                field_names.add(cond["key"])
        if field_names:
            await self.validate_fields_exist(
                collection_name, {k: None for k in field_names}
            )

        def to_field_condition(cond):
            return FieldCondition(key=cond["key"], match=MatchValue(**cond["match"]))

        filter_kwargs = {}
        for clause in ["must", "should", "must_not"]:
            if clause in filter_dict:
                filter_kwargs[clause] = [
                    to_field_condition(c) for c in filter_dict[clause]
                ]

        qdrant_filter = Filter(**filter_kwargs)
        selector = FilterSelector(filter=qdrant_filter)
        return await self.client.delete(collection_name, points_selector=selector)

    async def delete_collection(self, collection_name: str):
        await self.ensure_collection_exists(collection_name)
        return await self.client.delete_collection(collection_name)

    async def ensure_all_collections_exist(self):
        for collection in settings.QDRANT_COLLECTIONS:
            collection_name = collection["name"]
            if not await self.collection_exists(collection_name):
                await self.create_collection(collection_name)
                await self.create_payload_index(collection_name, "url")
                await self.create_payload_index(collection_name, "user_id")
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {collection_name}")

    async def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: str = "keyword"
    ) -> None:
        await self.ensure_collection_exists(collection_name)
        await self.validate_fields_exist(collection_name, {field_name: None})
        try:
            return await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as e:
            raise APIError(f"Failed to create payload index: {e}", status_code=500)

    async def has_data_for_user(self, collection_name: str, user_id: str) -> bool:
        qfilter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        result, _ = await self.client.scroll(
            collection_name=collection_name, scroll_filter=qfilter, limit=1
        )
        return len(result) > 0
