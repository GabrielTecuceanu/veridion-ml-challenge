from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, ScoredPoint

from src.config import QDRANT_COLLECTION, VECTOR_SEARCH_TOP_K
from src.indexing.embedding import encode_texts
from src.models.company import Company, CompanyMatch
from src.pipeline.metadata_filter import tag_missing_data

logger = logging.getLogger(__name__)


def _scored_point_to_match(point: ScoredPoint) -> CompanyMatch:
    """Convert a Qdrant ScoredPoint into a CompanyMatch stub.

    The full heuristic scoring happens in Stage 3; here we only populate
    the fields derivable from the Qdrant payload.
    """
    payload: dict[str, Any] = point.payload or {}

    # Reconstruct primary_naics dict from flat payload fields set by indexer
    primary_naics: dict | None = None
    pn_code = payload.get("primary_naics_code")
    pn_label = payload.get("primary_naics_label")
    if pn_code:
        primary_naics = {"code": pn_code, "label": pn_label or ""}

    # Reconstruct secondary_naics list
    sec_codes: list[str] = payload.get("secondary_naics_codes") or []
    secondary_naics = [{"code": c, "label": ""} for c in sec_codes]

    company = Company(
        id=str(point.id),
        name=payload.get("name", ""),
        description=payload.get("description", ""),
        country_code=payload.get("country_code"),
        is_public=payload.get("is_public"),
        employee_count=payload.get("employee_count"),
        revenue=payload.get("revenue"),
        year_founded=payload.get("year_founded"),
        business_model=payload.get("business_model") or [],
        primary_naics=primary_naics,
        secondary_naics=secondary_naics,
        tags=payload.get("tags") or [],
        raw=payload,
    )

    missing = tag_missing_data(payload)

    return CompanyMatch(
        company=company,
        score=0.0,  # filled by scorer
        vector_similarity=point.score,
        missing_data=missing,
        qualification_path=["stage2"],
    )


def search(
    query_text: str,
    client: QdrantClient,
    qdrant_filter: Filter | None = None,
    top_k: int = VECTOR_SEARCH_TOP_K,
) -> list[CompanyMatch]:
    """Stage 2: embed *query_text* and retrieve the top-k companies from Qdrant.

    Args:
        query_text:    Raw text to embed (can be the original query or a
                       rewritten variant from Stage 0.5).
        client:        Active QdrantClient instance.
        qdrant_filter: Optional payload filter from Stage 1.
        top_k:         Number of results to return.

    Returns:
        List of CompanyMatch objects ordered by descending vector similarity.
    """
    logger.debug("Embedding query for vector search: %r", query_text[:120])
    vectors = encode_texts([query_text])
    query_vector = vectors[0].tolist()

    results: list[ScoredPoint] = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    logger.info("Vector search returned %d candidates", len(results))
    return [_scored_point_to_match(p) for p in results]


def multi_query_search(
    query_texts: list[str],
    client: QdrantClient,
    qdrant_filter: Filter | None = None,
    top_k: int = VECTOR_SEARCH_TOP_K,
) -> list[CompanyMatch]:
    """Search with multiple query texts (Stage 0.5 variants) and union-merge.

    Deduplication keeps the highest vector_similarity score per company id.
    """
    seen: dict[str, CompanyMatch] = {}

    for text in query_texts:
        for match in search(text, client, qdrant_filter, top_k):
            cid = match.company.id
            if cid not in seen or (
                match.vector_similarity or 0.0
            ) > (seen[cid].vector_similarity or 0.0):
                seen[cid] = match

    merged = sorted(
        seen.values(),
        key=lambda m: m.vector_similarity or 0.0,
        reverse=True,
    )
    logger.info(
        "Multi-query union: %d unique companies from %d query variants",
        len(merged),
        len(query_texts),
    )
    return merged
