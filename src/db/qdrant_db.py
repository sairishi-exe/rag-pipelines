import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from src.config import COLPALI_DIM, QDRANT_COLLECTION, QDRANT_URL


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def init_collection(client: QdrantClient) -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=COLPALI_DIM,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM,
            ),
        ),
    )


def _point_id(pmcid: str, page_num: int) -> str:
    """Deterministic UUID per page — makes upserts idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{pmcid}:p{page_num:04d}"))


def is_already_embedded(client: QdrantClient, pmcid: str, expected_pages: int) -> bool:
    result = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="pmcid", match=MatchValue(value=pmcid))]
        ),
        exact=True,
    )
    return result.count == expected_pages


def upsert_page(
    client: QdrantClient,
    pmcid: str,
    page_num: int,
    embeddings: list[list[float]],
) -> None:
    """Store one page's full multi-vector (num_tokens x dim)."""
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=_point_id(pmcid, page_num),
                vector=embeddings,
                payload={"pmcid": pmcid, "page_num": page_num},
            )
        ],
    )


def query_pages(
    client: QdrantClient,
    query_embeddings: list[list[float]],
    top_p: int,
) -> list[dict]:
    """MaxSim search — Qdrant computes token-level max similarities internally."""
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embeddings,
        limit=top_p,
    )
    return [
        {"pmcid": p.payload["pmcid"], "page_num": p.payload["page_num"], "score": p.score}
        for p in results.points
    ]
