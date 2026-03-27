from __future__ import annotations

from src.config import (
    NAICS_SCORE_2DIGIT,
    NAICS_SCORE_4DIGIT,
    NAICS_SCORE_EXACT,
    NAICS_SCORE_NONE,
)


def naics_score(company_code: str | None, query_codes: list[str]) -> float:
    """Return the best NAICS alignment score between a company code and query codes.

    Scoring hierarchy:
        exact 6-digit match  → 1.0
        same 4-digit prefix  → 0.7
        same 2-digit sector  → 0.3
        no match             → 0.0
    """
    if not company_code or not query_codes:
        return NAICS_SCORE_NONE

    c = company_code.strip()
    best = NAICS_SCORE_NONE

    for q in query_codes:
        q = q.strip()
        if not q:
            continue

        if c == q:
            return NAICS_SCORE_EXACT  # can't do better

        if len(c) >= 4 and len(q) >= 4 and c[:4] == q[:4]:
            best = max(best, NAICS_SCORE_4DIGIT)
        elif len(c) >= 2 and len(q) >= 2 and c[:2] == q[:2]:
            best = max(best, NAICS_SCORE_2DIGIT)

    return best


def best_naics_score(
    primary_code: str | None,
    secondary_codes: list[str],
    query_codes: list[str],
) -> float:
    """Score using both primary and secondary NAICS; return the max."""
    scores = [naics_score(primary_code, query_codes)]
    for code in secondary_codes:
        scores.append(naics_score(code, query_codes))
    return max(scores)


def naics_prefix_filter(query_codes: list[str], prefix_len: int = 2) -> list[str]:
    """Extract unique N-digit prefixes from a list of NAICS codes.

    Useful for building Qdrant sector-level filters.
    """
    prefixes: set[str] = set()
    for code in query_codes:
        code = code.strip()
        if len(code) >= prefix_len:
            prefixes.add(code[:prefix_len])
    return sorted(prefixes)
