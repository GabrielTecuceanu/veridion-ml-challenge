from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from src.config import QDRANT_URL
from src.indexing.indexer import index
from src.models.company import CompanyMatch
from src.models.query_intent import QueryIntent
from src.pipeline.judge import run_judge
from src.pipeline.metadata_filter import build_filter
from src.pipeline.query_parser import parse_query
from src.pipeline.query_rewriter import rewrite_query
from src.pipeline.scorer import score_matches
from src.pipeline.vector_search import multi_query_search, search

logger = logging.getLogger(__name__)


@dataclass
class StageAttrition:
    after_stage1: int = 0
    after_stage2: int = 0
    after_stage3: int = 0
    after_stage4: int = 0


@dataclass
class PipelineResult:
    raw_query: str
    query_type: str
    parsed_intent: QueryIntent
    stage_attrition: StageAttrition
    qualified_companies: list[CompanyMatch]
    all_scored: list[CompanyMatch] = field(default_factory=list)


class Orchestrator:
    """Routes queries through the correct stage sequence based on query type.

    Type A (structured):   Stage 1 -> Stage 3 -> output
    Type B (hybrid):       Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 (borderline)
    Type C (reasoning):    Stage 0.5 -> Stage 1+2 -> Stage 3 -> Stage 4 (top 25)
    """

    def __init__(self, client: QdrantClient | None = None, ensure_index: bool = False):
        self._client = client or QdrantClient(location=QDRANT_URL)
        if ensure_index:
            index(self._client)

    # Stage 1: metadata-filtered scroll (Type A only - no vector search)

    def _stage1_scroll(self, qdrant_filter: Any) -> list[CompanyMatch]:
        """For Type A: retrieve all matching companies via scroll (no vector)."""
        from qdrant_client.models import Filter
        from src.config import QDRANT_COLLECTION
        from src.pipeline.metadata_filter import tag_missing_data
        from src.models.company import Company

        scroll_filter = qdrant_filter  # may be None -> returns all
        offset = None
        all_points = []

        while True:
            points, next_offset = self._client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        matches: list[CompanyMatch] = []
        for point in all_points:
            payload = point.payload or {}

            primary_naics: dict | None = None
            pn_code = payload.get("primary_naics_code")
            pn_label = payload.get("primary_naics_label")
            if pn_code:
                primary_naics = {"code": pn_code, "label": pn_label or ""}

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
                raw=payload,
            )
            missing = tag_missing_data(payload)
            matches.append(
                CompanyMatch(
                    company=company,
                    score=0.0,
                    missing_data=missing,
                    qualification_path=["stage1"],
                )
            )

        logger.info("Stage 1 scroll returned %d companies", len(matches))
        return matches

    def run(self, raw_query: str) -> PipelineResult:
        """Execute the full pipeline for a single query."""
        logger.info("=== Starting pipeline for: %r ===", raw_query)

        intent = parse_query(raw_query)
        query_type = intent.query_type
        attrition = StageAttrition()

        qdrant_filter = build_filter(intent)

        if query_type == "structured":
            # Type A: scroll with filter -> score -> no Judge
            matches = self._stage1_scroll(qdrant_filter)
            attrition.after_stage1 = len(matches)
            attrition.after_stage2 = len(matches)

            matches = score_matches(matches, intent)
            attrition.after_stage3 = len(matches)
            attrition.after_stage4 = len(matches)

        elif query_type == "hybrid":
            # Type B: filter + vector -> score -> Judge (borderline)
            matches = search(raw_query, self._client, qdrant_filter)
            attrition.after_stage1 = len(matches)
            attrition.after_stage2 = len(matches)
            for m in matches:
                if "stage1" not in m.qualification_path:
                    m.qualification_path.insert(0, "stage1")

            matches = score_matches(matches, intent)
            attrition.after_stage3 = len(matches)

            matches = run_judge(matches, intent, raw_query, query_type)
            attrition.after_stage4 = sum(1 for m in matches if m.score > 0)

        else:
            # Type C: rewrite -> filter + multi-vector -> score -> Judge (top 25)
            variants = rewrite_query(intent)
            all_queries = [raw_query] + variants

            matches = multi_query_search(all_queries, self._client, qdrant_filter)
            attrition.after_stage1 = len(matches)
            attrition.after_stage2 = len(matches)
            for m in matches:
                if "stage0.5" not in m.qualification_path:
                    m.qualification_path.insert(0, "stage0.5")
                if "stage1" not in m.qualification_path:
                    m.qualification_path.insert(1, "stage1")

            matches = score_matches(matches, intent)
            attrition.after_stage3 = len(matches)

            matches = run_judge(matches, intent, raw_query, query_type)
            attrition.after_stage4 = sum(1 for m in matches if m.score > 0)

        qualified = [m for m in matches if m.score > 0]

        logger.info(
            "Pipeline done: %d qualified out of %d scored",
            len(qualified),
            len(matches),
        )

        return PipelineResult(
            raw_query=raw_query,
            query_type=query_type,
            parsed_intent=intent,
            stage_attrition=attrition,
            qualified_companies=qualified,
            all_scored=matches,
        )
