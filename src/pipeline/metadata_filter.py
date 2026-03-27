from __future__ import annotations

import logging
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

from src.config import BUSINESS_MODEL_ALIASES
from src.models.query_intent import QueryIntent
from src.utils.naics import naics_prefix_filter

logger = logging.getLogger(__name__)


def _null_inclusive(
    field: str,
    has_field: str,
    condition: FieldCondition,
) -> Filter:
    """Return a Filter that matches records satisfying *condition* OR having
    the field absent (has_field == False).

    This implements the null-inclusive OR logic: never hard-exclude a company
    just because a field is missing — let Stage 3 penalise it instead.
    """
    return Filter(
        should=[
            condition,
            FieldCondition(key=has_field, match=MatchValue(value=False)),
        ]
    )


def _resolve_business_models(raw: list[str]) -> list[str]:
    """Expand shorthand aliases to canonical full strings."""
    result: list[str] = []
    for bm in raw:
        canonical = BUSINESS_MODEL_ALIASES.get(bm.lower(), bm)
        result.append(canonical)
    return result


def build_filter(intent: QueryIntent) -> Filter | None:
    """Stage 1: translate a QueryIntent into a Qdrant payload Filter.

    Design rules:
    - All clauses are AND-combined (must list).
    - Numeric filters use null-inclusive OR so missing-data companies survive.
    - Location filter: if resolved_countries is non-empty, match those codes OR
      null country — same null-inclusive pattern.
    - NAICS sector filter: 2-digit prefix match on naics_2digit field OR null.
    - Returns None when there are no applicable filters (pass-all).
    """
    must: list[Any] = []

    # --- Location -------------------------------------------------------
    if intent.location and intent.location.resolved_countries:
        countries = [c.upper() for c in intent.location.resolved_countries]
        must.append(
            Filter(
                should=[
                    FieldCondition(key="country_code", match=MatchAny(any=countries)),
                    # null country_code: company has no country recorded
                    FieldCondition(
                        key="country_code", match=MatchValue(value=None)
                    ),
                ]
            )
        )

    # --- Boolean filters ------------------------------------------------
    for field, value in intent.boolean_filters.items():
        must.append(FieldCondition(key=field, match=MatchValue(value=value)))

    # --- Business model -------------------------------------------------
    if intent.business_model_filter:
        canonical = _resolve_business_models(intent.business_model_filter)
        must.append(
            FieldCondition(key="business_model", match=MatchAny(any=canonical))
        )

    # --- NAICS sector (2-digit) ----------------------------------------
    if intent.naics_codes:
        sectors = naics_prefix_filter(intent.naics_codes, prefix_len=2)
        if sectors:
            must.append(
                Filter(
                    should=[
                        FieldCondition(
                            key="naics_2digit", match=MatchAny(any=sectors)
                        ),
                        FieldCondition(
                            key="naics_2digit", match=MatchValue(value=None)
                        ),
                    ]
                )
            )

    # --- Numeric filters ------------------------------------------------
    for nc in intent.numeric_filters:
        field = nc.field
        has_field = f"has_{field}"

        if nc.operator == "between" and nc.value2 is not None:
            range_cond = FieldCondition(
                key=field, range=Range(gte=nc.value, lte=nc.value2)
            )
        elif nc.operator == "gt":
            range_cond = FieldCondition(key=field, range=Range(gt=nc.value))
        elif nc.operator == "gte":
            range_cond = FieldCondition(key=field, range=Range(gte=nc.value))
        elif nc.operator == "lt":
            range_cond = FieldCondition(key=field, range=Range(lt=nc.value))
        elif nc.operator == "lte":
            range_cond = FieldCondition(key=field, range=Range(lte=nc.value))
        elif nc.operator == "eq":
            range_cond = FieldCondition(
                key=field, range=Range(gte=nc.value, lte=nc.value)
            )
        else:
            logger.warning("Unknown numeric operator %r — skipping", nc.operator)
            continue

        must.append(_null_inclusive(field, has_field, range_cond))

    if not must:
        return None

    return Filter(must=must)


def tag_missing_data(payload: dict[str, Any]) -> bool:
    """Return True if the company's payload has any null critical fields."""
    return not (
        payload.get("has_employee_count", False)
        and payload.get("has_revenue", False)
        and payload.get("country_code") is not None
    )
