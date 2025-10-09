from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from app.infrastructure.vectorstores.qdrant_config import qdrant_client
from app.api.exceptions import APIError
from app.config import settings

COLLECTION_DIM = 384
COLLECTION_DISTANCE = Distance.COSINE


def collection_exists(collection_name: str) -> bool:
    return qdrant_client.collection_exists(collection_name=collection_name)


def create_collection(collection_name: str):
    if collection_exists(collection_name):
        raise APIError(
            f"Collection '{collection_name}' already exists.", status_code=400
        )
    qdrant_client.create_collection(
        collection_name,
        vectors_config=VectorParams(size=COLLECTION_DIM, distance=COLLECTION_DISTANCE),
    )


def ensure_collection_exists(collection_name: str):
    if not collection_exists(collection_name):
        raise APIError(
            f"Collection '{collection_name}' does not exist.", status_code=404
        )


def validate_payload_fields(collection_name: str, payload: dict):
    """
    Ensure payload contains all required fields for the collection and no extra fields.
    Raise APIError if any required field is missing or if any extra field is present.
    """
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
            return  # Validation passed

    # If collection not found
    raise APIError(
        f"Collection '{collection_name}' is not defined in QDRANT_COLLECTIONS.",
        status_code=404,
    )


def validate_fields_exist(collection_name: str, field_dict: dict):
    """
    Ensure all fields in field_dict exist in the collection schema (required or optional).
    Raise APIError if any field does not exist.
    """
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
            return  # Validation passed
    raise APIError(
        f"Collection '{collection_name}' is not defined in QDRANT_COLLECTIONS.",
        status_code=404,
    )


def upsert_points(collection_name: str, points: list):
    ensure_collection_exists(collection_name)
    return qdrant_client.upsert(collection_name, wait=True, points=points)


def query_points(collection_name: str, vector: list, top: int = 5, filter: dict = None):
    ensure_collection_exists(collection_name)
    return qdrant_client.search(
        collection_name, query_vector=vector, limit=top, query_filter=filter
    )


def query_points_by_filter(
    collection_name: str, vector: list, top: int = 5, filter_dict: dict = None
):
    """
    Retrieve points from Qdrant collection using a dynamic filter.
    filter_dict example:
    {
        "must": [{"key": "city", "match": {"value": "London"}}],
        "should": [{"key": "color", "match": {"value": "red"}}],
        "must_not": [{"key": "status", "match": {"value": "inactive"}}]
    }
    """
    ensure_collection_exists(collection_name)
    if filter_dict:
        # Collect all field names from must, should, must_not
        field_names = set()
        for clause in ["must", "should", "must_not"]:
            for cond in filter_dict.get(clause, []):
                field_names.add(cond["key"])
        # Validate all fields
        if field_names:
            validate_fields_exist(collection_name, {k: None for k in field_names})

        def to_field_condition(cond):
            return FieldCondition(key=cond["key"], match=MatchValue(**cond["match"]))

        filter_kwargs = {}
        for clause in ["must", "should", "must_not"]:
            if clause in filter_dict:
                filter_kwargs[clause] = [
                    to_field_condition(c) for c in filter_dict[clause]
                ]

        qdrant_filter = Filter(**filter_kwargs)
    else:
        qdrant_filter = None

    return qdrant_client.search(
        collection_name, query_vector=vector, limit=top, query_filter=qdrant_filter
    )


def delete_points(collection_name: str, point_ids: list):
    ensure_collection_exists(collection_name)
    return qdrant_client.delete(collection_name, points_selector={"points": point_ids})


def delete_points_by_filter(collection_name: str, filter_dict: dict):
    """
    Delete points from Qdrant collection using a dynamic filter.
    filter_dict example:
    {
        "must": [{"key": "city", "match": {"value": "London"}}],
        "should": [{"key": "color", "match": {"value": "red"}}],
        "must_not": [{"key": "status", "match": {"value": "inactive"}}]
    }
    """
    # Collect all field names from must, should, must_not
    field_names = set()
    for clause in ["must", "should", "must_not"]:
        for cond in filter_dict.get(clause, []):
            field_names.add(cond["key"])
    # Validate all fields
    if field_names:
        validate_fields_exist(collection_name, {k: None for k in field_names})

    def to_field_condition(cond):
        return FieldCondition(key=cond["key"], match=MatchValue(**cond["match"]))

    filter_kwargs = {}
    for clause in ["must", "should", "must_not"]:
        if clause in filter_dict:
            filter_kwargs[clause] = [to_field_condition(c) for c in filter_dict[clause]]

    qdrant_filter = Filter(**filter_kwargs)
    selector = FilterSelector(filter=qdrant_filter)
    return qdrant_client.delete(collection_name, points_selector=selector)


def delete_collection(collection_name: str):
    ensure_collection_exists(collection_name)
    return qdrant_client.delete_collection(collection_name)


def create_payload_index(
    collection_name: str, field_name: str, field_schema: str = "keyword"
) -> None:
    """
    Create a payload index for a specific field in a Qdrant collection.
    Raises APIError if the collection does not exist or if index creation fails.
    """
    ensure_collection_exists(collection_name)
    validate_fields_exist(collection_name, {field_name: None})
    try:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
    except Exception as e:
        raise APIError(f"Failed to create payload index: {e}", status_code=500)
