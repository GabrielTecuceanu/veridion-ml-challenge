from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.config import (
    EMBEDDING_MODEL,
    INDEXER_BATCH_UPSERT,
    QDRANT_COLLECTION,
    QDRANT_DISTANCE,
    QDRANT_URL,
    QDRANT_VECTOR_SIZE,
)
from src.indexing.embedding import build_embedding_text, encode_texts
from src.indexing.normalizer import normalize_company

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
COMPANIES_FILE = DATA_DIR / "companies.jsonl"
NAICS_INVENTORY_FILE = DATA_DIR / "naics_inventory.json"


def _data_completeness(company_raw: dict[str, Any], company_obj: Any) -> float:
    """Pre-compute a data-completeness score [0, 1] at indexing time."""
    fields = [
        company_obj.country_code,
        company_obj.employee_count,
        company_obj.revenue,
        company_obj.year_founded,
        company_obj.primary_naics,
        company_obj.description,
        company_obj.business_model or None,
    ]
    present = sum(1 for f in fields if f is not None and f != [] and f != "")
    return round(present / len(fields), 4)


def _build_payload(company: Any) -> dict[str, Any]:
    """Build the Qdrant payload dict from a normalized Company."""
    primary_code: str | None = None
    if company.primary_naics:
        primary_code = company.primary_naics.get("code")

    secondary_codes: list[str] = []
    for n in company.secondary_naics:
        code = n.get("code")
        if code:
            secondary_codes.append(code)

    naics_2digit: str | None = primary_code[:2] if primary_code else None

    target_markets: list[str] = company.raw.get("target_markets") or []
    core_offerings: list[str] = company.raw.get("core_offerings") or []

    completeness = _data_completeness(company.raw, company)

    return {
        "id": company.id,
        "name": company.name,
        "description": company.description,
        "country_code": company.country_code,
        "is_public": company.is_public,
        "employee_count": company.employee_count,
        "revenue": company.revenue,
        "year_founded": company.year_founded,
        "business_model": company.business_model,
        "primary_naics_code": primary_code,
        "primary_naics_label": (company.primary_naics or {}).get("label"),
        "secondary_naics_codes": secondary_codes,
        "naics_2digit": naics_2digit,
        "target_markets": target_markets,
        "core_offerings": core_offerings,
        "has_employee_count": company.employee_count is not None,
        "has_revenue": company.revenue is not None,
        "has_country_code": company.country_code is not None,
        "data_completeness": completeness,
    }


def _build_naics_inventory(companies: list[Any]) -> dict[str, str]:
    """Collect all NAICS code -> label mappings present in the dataset."""
    inventory: dict[str, str] = {}
    for c in companies:
        if c.primary_naics:
            code = c.primary_naics.get("code")
            label = c.primary_naics.get("label", "")
            if code:
                inventory[code] = label
        for n in c.secondary_naics:
            code = n.get("code")
            label = n.get("label", "")
            if code:
                inventory[code] = label
    return dict(sorted(inventory.items()))


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        distance = Distance[QDRANT_DISTANCE]
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=distance),
        )
        logger.info("Created Qdrant collection '%s'", QDRANT_COLLECTION)
    else:
        logger.info("Qdrant collection '%s' already exists - skipping creation", QDRANT_COLLECTION)


def load_companies(path: Path = COMPANIES_FILE) -> list[Any]:
    raw_records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    logger.info("Loaded %d raw records from %s", len(raw_records), path)

    companies = []
    for record in raw_records:
        try:
            companies.append(normalize_company(record))
        except Exception as exc:
            logger.warning("Skipping record (normalize error): %s", exc)
    logger.info("Normalized %d companies", len(companies))
    return companies


def index(client: QdrantClient | None = None) -> None:
    """Main entry point: load -> normalize -> embed -> upsert."""
    if client is None:
        client = QdrantClient(location=QDRANT_URL)

    _ensure_collection(client)

    companies = load_companies()

    inventory = _build_naics_inventory(companies)
    NAICS_INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NAICS_INVENTORY_FILE.open("w") as f:
        json.dump(inventory, f, indent=2)
    logger.info("Wrote NAICS inventory (%d codes) to %s", len(inventory), NAICS_INVENTORY_FILE)

    texts = [build_embedding_text(c, c.raw) for c in companies]
    logger.info("Encoding %d companies with %s ...", len(companies), EMBEDDING_MODEL)
    embeddings = encode_texts(texts)

    total_batches = math.ceil(len(companies) / INDEXER_BATCH_UPSERT)
    with tqdm(total=len(companies), desc="Upserting to Qdrant") as pbar:
        for batch_idx in range(total_batches):
            start = batch_idx * INDEXER_BATCH_UPSERT
            end = start + INDEXER_BATCH_UPSERT
            batch_companies = companies[start:end]
            batch_embeddings = embeddings[start:end]

            points = []
            for i, (company, vector) in enumerate(zip(batch_companies, batch_embeddings)):
                payload = _build_payload(company)
                point_id = start + i
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )

            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            pbar.update(len(batch_companies))

    logger.info("Indexing complete. %d companies upserted.", len(companies))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    index()
