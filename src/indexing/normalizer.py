from __future__ import annotations

import ast
import logging
from typing import Any

from src.config import BUSINESS_MODEL_ALIASES, REVENUE_SANITY_CAP
from src.models.company import Company

logger = logging.getLogger(__name__)


def _parse_naics(raw: Any) -> dict[str, Any] | None:
    """Parse a NAICS value that may be a dict, a Python dict-literal string, or None."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            logger.debug("Could not parse NAICS string: %r", raw)
    return None


def _parse_address(raw: Any) -> dict[str, Any]:
    """Parse address field which may be a dict or a Python dict-literal string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            logger.debug("Could not parse address string: %r", raw)
    return {}


def _normalize_country_code(code: Any) -> str | None:
    """Uppercase ISO-2 country code, or None if missing."""
    if not code or not isinstance(code, str):
        return None
    stripped = code.strip()
    if not stripped:
        return None
    return stripped.upper()


def _cap_revenue(revenue: Any) -> float | None:
    """Return revenue as float capped at REVENUE_SANITY_CAP, or None."""
    if revenue is None:
        return None
    try:
        value = float(revenue)
    except (TypeError, ValueError):
        return None
    return min(value, REVENUE_SANITY_CAP)


def _parse_secondary_naics(raw: Any) -> list[dict[str, Any]]:
    """Parse secondary_naics which may be None, a list, or a string."""
    if raw is None:
        return []
    if isinstance(raw, list):
        results = []
        for item in raw:
            parsed = _parse_naics(item)
            if parsed:
                results.append(parsed)
        return results
    parsed = _parse_naics(raw)
    return [parsed] if parsed else []


def normalize_company(raw: dict[str, Any]) -> Company:
    """Normalize a raw JSONL record into a Company model.

    Handles:
    - NAICS dict-literal strings -> dicts
    - country_code lowercase -> uppercase
    - revenue float cap at $10T
    - address string -> dict
    - employee_count / year_founded -> int
    """
    address = _parse_address(raw.get("address"))
    country_code = _normalize_country_code(address.get("country_code"))

    primary_naics = _parse_naics(raw.get("primary_naics"))
    secondary_naics = _parse_secondary_naics(raw.get("secondary_naics"))

    revenue = _cap_revenue(raw.get("revenue"))

    employee_count = raw.get("employee_count")
    if employee_count is not None:
        try:
            employee_count = int(float(employee_count))
        except (TypeError, ValueError):
            employee_count = None

    year_founded = raw.get("year_founded")
    if year_founded is not None:
        try:
            year_founded = int(float(year_founded))
        except (TypeError, ValueError):
            year_founded = None

    raw_bm: list[str] = raw.get("business_model") or []
    if not isinstance(raw_bm, list):
        raw_bm = []
    business_model: list[str] = [
        BUSINESS_MODEL_ALIASES.get(v.lower(), v) for v in raw_bm if isinstance(v, str)
    ]

    # website is the unique identifier in the dataset
    company_id = (raw.get("website") or "").strip()
    description: str = (raw.get("description") or "").strip()

    return Company(
        id=company_id,
        name=(raw.get("operational_name") or raw.get("website") or "").strip(),
        description=description,
        country_code=country_code,
        is_public=raw.get("is_public"),
        employee_count=employee_count,
        revenue=revenue,
        year_founded=year_founded,
        business_model=business_model,
        primary_naics=primary_naics,
        secondary_naics=secondary_naics,
        tags=[],
        raw=raw,
    )
