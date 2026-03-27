from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from src.config import REWRITE_NUM_VARIANTS
from src.models.query_intent import QueryIntent
from src.utils.llm_client import structured_completion

logger = logging.getLogger(__name__)


class _RewriteResponse(BaseModel):
    queries: list[str] = Field(
        description="Alternative supplier-perspective queries",
        min_length=1,
    )


_SYSTEM_PROMPT = """\
You are an expert at reformulating supply-chain search queries.
Given an original ecosystem query (which describes what a buyer needs),
rewrite it from the *supplier's* perspective so that a company matching
the description would actually use these words to describe itself.

Rules:
- Write each variant as a short, self-contained search query.
- Each variant must probe a distinct semantic neighbourhood
  (e.g. one focuses on materials, one on processes, one on certifications).
- Do NOT just add synonyms — write genuinely different queries.
- Avoid mentioning the end-customer / buyer in the rewritten queries.
- Return exactly the number of queries requested.
"""


def rewrite_query(intent: QueryIntent, n: int = REWRITE_NUM_VARIANTS) -> list[str]:
    """Stage 0.5: expand a Type C query into supplier-perspective variants.

    Each variant is a standalone query string that will be embedded
    independently. Results are union-merged by the orchestrator with
    max-score deduplication.

    Returns a list of *n* alternative query strings.
    """
    if not intent.ecosystem_role and not intent.target_beneficiary:
        logger.warning(
            "rewrite_query called on intent without ecosystem_role/target_beneficiary"
        )

    original_parts = []
    if intent.industry_keywords:
        original_parts.append("industry: " + ", ".join(intent.industry_keywords))
    if intent.ecosystem_role:
        original_parts.append(f"role: {intent.ecosystem_role}")
    if intent.target_beneficiary:
        original_parts.append(f"serving: {intent.target_beneficiary}")
    if intent.semantic_criteria:
        original_parts.append(f"criteria: {intent.semantic_criteria}")

    original_summary = "; ".join(original_parts) if original_parts else "(no details)"

    prompt = (
        f"Original query intent: {original_summary}\n\n"
        f"Generate {n} alternative search queries written from the supplier's "
        f"perspective. Each query should describe what a qualifying company "
        f"does, not what the buyer needs."
    )

    response = structured_completion(
        prompt=prompt,
        response_model=_RewriteResponse,
        system=_SYSTEM_PROMPT,
    )

    queries = response.queries[:n]
    logger.info("Rewritten %d supplier-perspective queries", len(queries))
    for i, q in enumerate(queries, 1):
        logger.debug("  variant %d: %r", i, q)
    return queries
