from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from src.pipeline.orchestrator import Orchestrator, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

QUERIES_FILE = Path(__file__).parent.parent / "eval" / "queries.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _slug(query: str) -> str:
    """Convert a query string into a safe filename slug."""
    slug = query.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")[:80]


def _serialize_result(result: PipelineResult) -> dict:
    """Convert a PipelineResult to a JSON-serialisable dict."""
    return {
        "query": result.raw_query,
        "query_type": result.query_type,
        "parsed_intent": result.parsed_intent.model_dump(),
        "stage_attrition": {
            "after_stage1": result.stage_attrition.after_stage1,
            "after_stage2": result.stage_attrition.after_stage2,
            "after_stage3": result.stage_attrition.after_stage3,
            "after_stage4": result.stage_attrition.after_stage4,
        },
        "qualified_companies": [
            {
                "rank": m.rank,
                "confidence": m.confidence,
                "id": m.company.id,
                "name": m.company.name,
                "country_code": m.company.country_code,
                "score": m.score,
                "vector_similarity": m.vector_similarity,
                "naics_alignment": m.naics_alignment,
                "keyword_overlap": m.keyword_overlap,
                "constraint_satisfaction": m.constraint_satisfaction,
                "data_completeness": m.data_completeness,
                "missing_data": m.missing_data,
                "qualification_path": m.qualification_path,
                "matched_criteria": m.matched_criteria,
                "failed_criteria": m.failed_criteria,
                "reasoning": m.reasoning,
            }
            for m in result.qualified_companies
        ],
    }


def run_all(queries: list[str], ensure_index: bool = True) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    orchestrator = Orchestrator(ensure_index=ensure_index)

    for query in queries:
        logger.info("Running query: %r", query)
        try:
            result = orchestrator.run(query)
        except Exception as exc:
            logger.error("Query failed: %r - %s", query, exc)
            continue

        slug = _slug(query)
        out_path = RESULTS_DIR / f"{slug}.json"
        payload = _serialize_result(result)
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "Wrote %d qualified companies -> %s",
            len(result.qualified_companies),
            out_path,
        )


def main() -> None:
    with QUERIES_FILE.open() as f:
        data = json.load(f)

    queries: list[str] = [
        q if isinstance(q, str) else q["query"] for q in data.get("queries", [])
    ]

    if not queries:
        logger.error("No queries found in %s", QUERIES_FILE)
        sys.exit(1)

    run_all(queries)


if __name__ == "__main__":
    main()
