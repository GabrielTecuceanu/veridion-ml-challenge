# Intent-Aware Company Qualification System

## Overview

A cascading funnel pipeline that qualifies companies from a dataset against natural-language queries. The system routes each query through the minimum number of stages needed to answer it accurately, avoiding expensive LLM calls where simpler signals suffice.

Design priorities: **Accuracy > Scalability > Speed > Simplicity**

---

## 1. Data Analysis

Before writing any code, I explored the dataset to understand what I was working with.

**Field completeness:**

- `description`: 100% present - the most reliable signal
- `revenue`: ~80% present, but some values exceeded $10 trillion (data error)
- `employee_count`: ~61% present - the largest uncertainty source
- `country_code`: ~87% present, already ISO-2 lowercase
- `primary_naics`: ~100% present, but ~13% stored as Python dict-literal strings instead of parsed dicts
- `secondary_naics`: only ~2.3% coverage - not a reliable signal
- `business_model`: canonical full strings (e.g. `"Business-to-Business"`), inconsistent with common shorthand like `"B2B"`

**Industry distribution:**

- Only 4 companies in NAICS sector 48-49 (logistics/transportation) - queries like "logistics in Romania" will correctly return very small result sets
- Tech, manufacturing, and services dominate

**Key insight:** missing data is not random noise - it is structurally correlated. Companies with null `employee_count` are often smaller or private. Hard-excluding them would silently drop valid matches.

---

## 2. Assumptions

All assumptions are grounded in what the data exploration revealed.

- **`primary_naics` dict-strings**: 64/477 records store this field as a Python dict-literal string. Normalized at ingestion with `ast.literal_eval()`. No downstream code ever sees the raw form.
- **`country_code` already structured**: the field is lowercase ISO-2 directly in the JSON. No geocoding or resolver needed - just uppercase at normalization time.
- **`business_model` full strings only**: the dataset uses `"Business-to-Business"`, not `"B2B"`. A `BUSINESS_MODEL_ALIASES` map in `config.py` handles shorthand in queries.
- **Revenue cap at $10T**: values above `$10,000,000,000,000` are clearly erroneous. Capped at ingestion via `REVENUE_SANITY_CAP`.
- **Null fields = uncertain, not absent**: a company without `employee_count` is not necessarily small. Null fields score 0.5 (uncertain) throughout scoring, never 0.0.
- **Small logistics dataset is correct**: 4 logistics companies in the dataset is accurate, not a bug. The pipeline does not inflate results to meet an expected count.

---

## 3. Goals and Non-Goals

**Goals:**

- Correctly qualify companies for all 12 test queries, including supply-chain and ecosystem queries
- Cheaper and faster than sending every company to an LLM
- Gracefully handle missing data without dropping valid candidates
- Produce interpretable output with per-company reasoning and matched/failed criteria
- Architecture that scales beyond 477 companies without redesign

**Non-Goals:**

- Real-time / sub-second latency (indexing is offline; query latency of a few seconds is acceptable)
- Full coverage of every possible query type imaginable
- Fine-tuned models or training data
- Beating a GPT-4-class model on accuracy at any cost

---

## 4. Approach

### 4.1 Why Not the Baselines?

**Baseline A (LLM per company):** Sending all 477 companies to an LLM for every query means ~477 serial LLM calls. At even 1 second per call that is ~8 minutes per query. Costs scale linearly with dataset size. The same expensive treatment is applied to a simple filter query ("public companies in Germany") as to a complex ecosystem query.

**Baseline B (pure embedding similarity):** Fast and cheap, but similarity != relevance. A query for "packaging suppliers for cosmetics brands" retrieves cosmetics _brands_ because the embedding space conflates buyer and supplier. No structured filters, no NAICS alignment, no reasoning about intent.

**The solution:** a cascading funnel that routes each query through only the stages it needs.

---

### 4.2 Architecture: Cascading Funnel with Adaptive Routing

```
Query
  |
Stage 0: Query Parser (LLM -> QueryIntent)
  |
  +-- Type A (structured) --> Stage 1 --> Stage 3 --> Output
  |
  +-- Type B (hybrid)     --> Stage 1 --> Stage 2 --> Stage 3 --> Stage 4 (borderline) --> Output
  |
  +-- Type C (reasoning)  --> Stage 0.5 --> Stage 1+2 --> Stage 3 --> Stage 4 (top 25) --> Output
```

Each stage reduces the candidate pool before the next, more expensive stage.

---

### Stage 0 - Query Understanding

**File:** `src/pipeline/query_parser.py`

The raw query is sent to Qwen 3 8B (via Ollama + `instructor`) to produce a `QueryIntent` Pydantic object. This includes:

- `query_type`: `"structured"` / `"hybrid"` / `"reasoning"` - determines routing
- `naics_codes`: LLM-predicted 6-digit NAICS codes (max 3)
- `location.resolved_countries`: region names resolved to ISO-2 lists ("Scandinavia" -> `["SE","NO","DK","FI"]`)
- `numeric_filters`: structured constraints on `employee_count`, `revenue`, `year_founded`
- `business_model_filter`: canonical full strings only
- `semantic_criteria`: anything that cannot be expressed as a hard filter - passed to the Judge
- `ecosystem_role` + `target_beneficiary`: set only for supply-chain queries, triggers Stage 0.5

**Guard:** if `semantic_criteria` is non-empty, `query_type` is forced to at least `"hybrid"` regardless of the LLM's classification. This prevents soft criteria from being silently dropped.

Temperature is 0.0 throughout for determinism.

---

### Stage 0.5 - Query Rewriting (Type C only)

**File:** `src/pipeline/query_rewriter.py`

Supply-chain queries fail with pure embedding search because the query describes the buyer's need, but the relevant companies describe themselves from the supplier's perspective. "Suppliers of eco-friendly packaging for cosmetics brands" will retrieve cosmetics brands, not packaging manufacturers.

The rewriter generates 4 alternative queries written from the supplier's perspective, each probing a distinct semantic neighborhood:

```
Original: "Find suppliers of eco-friendly packaging for cosmetics brands"
Variant 1: "Sustainable packaging materials manufacturer"
Variant 2: "Biodegradable packaging for beauty and personal care"
Variant 3: "Recycled and eco-certified packaging solutions"
Variant 4: "Primary packaging supplier for cosmetics and skincare"
```

All 5 queries (original + 4 variants) are embedded independently. Results are union-merged with max-score deduplication per company ID.

Why separate queries instead of synonym expansion? Each variant occupies a different region of embedding space. A single expanded query averages these regions and loses the distinctions.

---

### Stage 1 - Metadata Filtering

**File:** `src/pipeline/metadata_filter.py`

Translates `QueryIntent` into a Qdrant payload filter. The critical design principle here is **null-inclusive OR**:

```python
# Instead of: revenue > 50_000_000
# We build:   revenue > 50_000_000 OR has_revenue == False
```

Companies with null fields pass the filter and are penalized with `NULL_FIELD_SCORE = 0.5` in Stage 3. They are never hard-excluded - the Judge gets to decide.

Boolean fields like `is_public` are the exception: if the query asks for public companies, private ones are hard-excluded. The intent is unambiguous.

---

### Stage 2 - Vector Search

**File:** `src/pipeline/vector_search.py`

Embedding model: `intfloat/multilingual-e5-large` (1024-dim, ONNX via fastembed). Top 50 results per query vector.

Company texts are built with labeled field prefixes at index time:

```
Company: Meridian Logistics GmbH
Industry: Freight Transportation Arrangement (488510)
Description: Full-service freight forwarding...
Markets served: automotive, manufacturing
Core offerings: freight forwarding; customs brokerage; warehousing
Business model: Business-to-Business
Country: DE
```

Labeled prefixes help the embedding model distinguish field types. Raw concatenation loses this structure.

Type A queries skip Stage 2 entirely - they use a Qdrant scroll with the metadata filter from Stage 1.

---

### Stage 3 - Heuristic Scoring

**File:** `src/pipeline/scorer.py`

Weighted composite score over five signals:

| Signal                  | Weight | Rationale                              |
| ----------------------- | ------ | -------------------------------------- |
| Vector similarity       | 0.30   | Primary semantic signal                |
| NAICS alignment         | 0.25   | Industry precision; complements vector |
| Keyword overlap         | 0.20   | Fast lexical check on industry terms   |
| Constraint satisfaction | 0.20   | Hard filter compliance                 |
| Data completeness       | 0.05   | Mild tiebreaker                        |

**NAICS hierarchical scoring:**

- Exact 6-digit match: 1.0
- Same 4-digit prefix: 0.7
- Same 2-digit sector: 0.3
- No match: 0.0

Both `primary_naics` and `secondary_naics` are checked; the max score is used.

**Why these weights?** Vector similarity is the broadest signal but can confuse buyer/supplier roles. NAICS + keyword together (0.45) provide structured industry grounding. Constraint satisfaction (0.20) ensures hard filters are reflected in ranking even after the null-inclusive filter lets borderline cases through. Data completeness (0.05) is a mild tiebreaker, not a primary signal.

Null fields score `0.5` (uncertain), not `0.0`. A company without `employee_count` is not penalized - it is simply uncertain.

---

### Stage 4 - LLM Judge

**File:** `src/pipeline/judge.py`

The Judge is invoked selectively:

- **Type A:** skipped entirely
- **Type B:** top 10 companies in the ambiguous score zone `[0.40, 0.70]`
- **Type C:** top 25 companies regardless of score

Companies are batched 5 per prompt. The system prompt includes calibration examples (true positive, true negative, borderline) to anchor the model's decision boundary.

Post-verdict validation: `matched_criteria` are checked against the actual profile payload. Any criterion that cannot be grounded in a real profile field is dropped. The verdict itself stands.

Companies the Judge disqualifies get their score zeroed and sink to the bottom of the ranked list.

**Why batch size 5?** Smaller batches lose cross-company context that helps the model calibrate borderline cases. Larger batches increase prompt length and token cost without meaningfully improving accuracy for an 8B model.

**Why top-10 borderline for Type B?** High-scoring Type B companies (> 0.70) already satisfy filters and semantic criteria well - calling the Judge on them wastes tokens. Low-scoring ones (< 0.40) are almost certainly not matches. The Judge's value is in resolving ambiguity in the 0.40-0.70 zone.

---

### Cost Comparison

For a Type B query on 477 companies:

| Approach                     | LLM calls | Companies per call |
| ---------------------------- | --------- | ------------------ |
| Baseline A (LLM per company) | 477       | 1                  |
| This system                  | ~2        | 5 (batched)        |

Stage 0 costs 1 LLM call (query parsing). Stage 4 costs at most `ceil(10 / 5) = 2` batched calls. Total: **3 LLM calls** instead of 477. For Type C it is at most `ceil(25 / 5) + 1 = 6` calls, plus 1 for Stage 0.5 rewriting. The LLM is reserved for decisions that genuinely require language understanding.

---

## 5. Tradeoffs

**Optimized for:** accuracy on the given 12 queries, interpretability of output, and correctness on missing data.

**Intentional tradeoffs:**

- **Qwen 3 8B over a larger model:** runs locally, zero API cost, acceptable accuracy for structured extraction and borderline judgment. Larger models would improve edge cases but require cloud API calls.
- **Null-inclusive filtering increases candidate set size:** passing null-field companies through Stage 1 means Stage 3 scores more companies. The tradeoff is higher recall at the cost of slightly more Stage 3 compute. Given Stage 3 is pure Python arithmetic, this is cheap.
- **Fixed heuristic weights over learned weights:** interpretable and stable, but not adaptive. A dataset with different field distributions would need weight re-tuning.
- **Stage 0.5 adds one LLM call per Type C query:** necessary for supply-chain queries; skipped for all others.
- **Judge temperature 0:** deterministic results at the cost of diversity. For this task, consistency is more valuable than exploration.

---

## 6. Error Analysis

**Where the system works well:**

- Structured queries with complete data (location, business model, employee count all present)
- Queries with clear NAICS alignment
- Supply-chain queries with enough description text for embedding to anchor on

**Where it struggles:**

**Case 1: Logistics in Romania**
Only 4 logistics companies (NAICS 48-49) exist in the dataset, and not all are in Romania. The system correctly returns a small result set - but a user expecting 20+ results might assume the pipeline is broken. This is a data coverage problem, not a pipeline problem. The system does not hallucinate matches to fill a quota.

**Case 2: "Fast-growing fintech competing with traditional banks"**
"Fast-growing" is not a field. "Competing with traditional banks" requires inferring a business context not present in most descriptions. The Judge can reason about this for top candidates, but if the vector search doesn't surface the right companies in the first place, the Judge never sees them. This is the hardest class of query - the pipeline's recall ceiling is determined by Stage 2.

**Case 3: Companies without descriptions**
Description is 100% present in this dataset, so this is theoretical - but the embedding quality degrades significantly if description is missing. A company known only by its NAICS code and name has a weak vector representation.

**Case 4: NAICS misclassification in the source data**
If a company's NAICS code is wrong (e.g., a SaaS company classified under a manufacturing code), NAICS alignment scores 0.0 even though vector + keyword might score well. Weight 0.25 on NAICS means a wrong code costs the company ~0.25 of its potential score - enough to push it below the Judge threshold in Type B.

---

## 7. Scaling

**Current (477 companies):** everything fits in Qdrant in-memory. Query-time cost is dominated by embedding the query and the Stage 4 LLM calls.

**100K companies:**

- Enable Qdrant HNSW (already the default). Vector search remains O(log N).
- Stage 1 (scroll) becomes the bottleneck for Type A queries - switch to filtered vector search for all query types.
- Increase `ef_construct` from 128 to 256 for better HNSW recall.
- Qdrant memory footprint: ~400MB for 100K x 1024-dim float32 vectors. Still fits on a single machine.
- No architectural changes needed.

**10M companies:**

- Qdrant sharding, partitioned by region or NAICS sector (natural query boundaries).
- Enable Scalar Quantization (8-bit): 4x memory reduction with ~1% accuracy loss.
- Replace fastembed CPU inference with a GPU embedding service (Hugging Face TEI) for parallel batch encoding at indexing time.
- Cache `QueryIntent` for repeated or similar queries - Stage 0 adds ~1-2 seconds and the intent is deterministic.
- The funnel architecture means query-time LLM calls stay constant (3-6 per query) regardless of dataset size. That is the key property that makes this design scalable.

---

## 8. Failure Modes

**Confident but wrong:**

- **Supplier/buyer confusion in Type B:** a cosmetics company can have high vector similarity to a packaging query without Stage 0.5 rewriting. If classified as Type B instead of Type C, the rewriting step is skipped and the Judge sees buyer-side companies with scores in the 0.5-0.7 range.

- **LLM hallucination in the Judge:** Qwen 3 8B occasionally cites a criterion that is not in the profile (e.g., claims a company is "Series B funded" when that is not mentioned). The post-verdict validation catches obvious cases, but subtle ones can slip through.

- **Query misclassification:** if the LLM classifies a reasoning query as "hybrid", Stage 0.5 is skipped and the multi-vector search is not run. The `semantic_criteria` guard prevents structured queries from being under-classified, but Type B vs. Type C is harder to enforce programmatically.

- **NAICS code hallucination:** the query parser predicts NAICS codes from query text. If it predicts wrong codes, the NAICS alignment component penalizes all valid companies. This is visible in the output (`naics_alignment` score per company) and can be debugged by inspecting `parsed_intent.naics_codes`.

**What to monitor in production:**

- Distribution of `query_type` classifications - a shift toward more Type A for complex queries signals model drift
- Average Judge confidence - if median confidence drops below 0.6, the Judge is uncertain and the score threshold may need tuning
- Stage attrition counts - if Stage 1 passes fewer than 5 companies to Stage 2, the metadata filters may be over-constraining
- Null-field rate in candidates that reach Stage 4 - a spike means the real data is getting sparser

---

## 9. File Structure

```
veridion-ml-challenge/
|- data/
|  |- companies.jsonl          # 477 company records
|  `- naics_inventory.json     # 105 NAICS codes, auto-generated at index time
|- eval/
|  `- queries.json             # 12 test queries
|- results/                    # per-query JSON output
|- src/
|  |- config.py                # all tunable constants
|  |- main.py                  # CLI: run all 12 queries
|  |- models/
|  |  |- company.py            # Company, CompanyMatch
|  |  |- query_intent.py       # QueryIntent, LocationConstraint, NumericConstraint
|  |  `- judge_verdict.py      # JudgeVerdict
|  |- pipeline/
|  |  |- orchestrator.py       # routes query type A/B/C through stages
|  |  |- query_parser.py       # Stage 0: LLM -> QueryIntent
|  |  |- query_rewriter.py     # Stage 0.5: supplier-perspective query expansion
|  |  |- metadata_filter.py    # Stage 1: Qdrant filter builder
|  |  |- vector_search.py      # Stage 2: embed + retrieve
|  |  |- scorer.py             # Stage 3: heuristic weighted scoring
|  |  `- judge.py              # Stage 4: batched LLM Judge
|  |- indexing/
|  |  |- indexer.py            # JSONL -> normalize -> embed -> upsert
|  |  |- embedding.py          # build_embedding_text() + batch encode
|  |  `- normalizer.py         # NAICS str->dict, revenue cap, country_code
|  `- utils/
|     |- naics.py              # NAICS hierarchy prefix matching
|     `- llm_client.py         # Ollama client with retry + instructor
|- explore_data.py             # dataset statistics and field completeness
|- Dockerfile
|- docker-compose.yml
`- requirements.txt
```

All tunable parameters live in `config.py`. No magic numbers in pipeline code.

---

## 10. Running with Docker

Requires Docker and Docker Compose. The first run downloads `qwen3:8b` (~6GB) into a named volume.

```bash
docker compose up --build
```

On subsequent runs the model and Qdrant data are reused from volumes - no re-download, no re-indexing:

```bash
docker compose up
```

Results are written to `./results/` on the host.

To run a custom query instead of the 12 test queries:

```bash
QUERY="your query here" docker compose up app
```

Without a GPU, Ollama falls back to CPU inference (correct results, slower LLM calls).

---

## 11. Time Breakdown

| Phase                                  | Time          |
| -------------------------------------- | ------------- |
| Data exploration and research          | ~4 hours      |
| Writing the spec and architecture plan | ~1 hour       |
| Building the pipeline                  | ~10 hours     |
| Docker, documentation, cleanup         | ~2 hours      |
| **Total**                              | **~17 hours** |

The majority of implementation time went into the metadata filter builder (null-inclusive OR logic across all field types) and the scoring system (getting NAICS hierarchical scoring and constraint satisfaction to correctly handle nulls). The architecture itself was settled early - the exploration phase revealed enough about the data to design the funnel before writing a line of pipeline code.
