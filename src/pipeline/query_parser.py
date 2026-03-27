from __future__ import annotations

import logging

from src.models.query_intent import QueryIntent
from src.utils.llm_client import structured_completion

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert business intelligence analyst.
Your task: parse a natural-language company search query into a structured JSON object.

Guidelines:
- query_type:
    "structured"  → query uses only hard filters (location, size, industry, business model, founding year)
    "hybrid"      → mix of hard filters + soft semantic criteria
    "reasoning"   → supply-chain / ecosystem / inferred-role queries that require logical deduction
- naics_codes: predict the most relevant 6-digit NAICS codes (top 3 max). Leave empty if truly unclear.
- location.resolved_countries: resolve region names to ISO-2 codes.
  Examples: "Scandinavia" → ["SE","NO","DK","FI"], "Benelux" → ["BE","NL","LU"],
  "DACH" → ["DE","AT","CH"], "CEE" → ["CZ","PL","HU","RO","SK","BG","HR","SI","EE","LV","LT"]
- numeric_filters: extract employee_count / revenue / year_founded constraints.
  Revenue values like "$50M" → 50000000, "$1B" → 1000000000.
- business_model_filter: use FULL strings only:
  "Business-to-Business", "Business-to-Consumer", "Business-to-Government",
  "Software-as-a-Service", "Marketplace", "Wholesale", "Retail", "Enterprise",
  "Manufacturing", "Service Provider"
- semantic_criteria: any criterion that cannot be expressed as a hard filter
  (e.g. "specialises in cold-chain logistics", "serves luxury brands").
- ecosystem_role + target_beneficiary: fill ONLY for supply-chain queries.
  e.g. "suppliers of eco-friendly packaging for cosmetics brands"
       → ecosystem_role="supplier", target_beneficiary="cosmetics brands"
"""


def parse_query(raw_query: str) -> QueryIntent:
    """Stage 0: parse a natural-language query into a QueryIntent.

    Uses instructor-enforced structured output via Ollama. Applies the
    semantic_criteria guard (forces query_type ≥ "hybrid") via the
    QueryIntent model_validator.
    """
    logger.info("Parsing query: %r", raw_query)

    prompt = f'Parse this company search query:\n\n"{raw_query}"'
    intent = structured_completion(
        prompt=prompt,
        response_model=QueryIntent,
        system=_SYSTEM_PROMPT,
    )

    logger.info(
        "Parsed intent: type=%s naics=%s location=%s",
        intent.query_type,
        intent.naics_codes,
        intent.location,
    )
    return intent
