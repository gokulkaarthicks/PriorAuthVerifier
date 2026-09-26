# PriorAuthVerifier

Prior-authorization readiness checker for patient charts and payer policies.

## Overview

Given a patient chart PDF and a payer policy PDF, the service:

1. Detects the planned procedure from the patient chart.
2. Retrieves governing policy criteria for that procedure.
3. Reports whether matching policy criteria were found
   (`matching_policy_criteria_found`).
4. Evaluates patient evidence against each criterion.
5. Returns a deterministic readiness result.

## Architecture

The system is a request-scoped, in-memory pipeline. There is no vector database
and no persistent retrieval index. Each analysis parses the patient PDF once
and the policy PDF once, builds logical sections, then runs three retrieval
paths:

| Stage | Purpose |
|---|---|
| Procedure discovery | Find the next planned procedure |
| Policy retrieval | Find governing criteria for that procedure |
| Evidence verification | Check patient chart support for each criterion |

Readiness is computed deterministically from the criteria tree and evidence
statuses.

Tech stack: FastAPI + Uvicorn, PyMuPDF, Mistral OCR (when needed), BAAI/bge-m3,
BAAI/bge-reranker-v2-m3, rank-bm25, gpt-6-luna (procedure), gpt-6-sol (policy +
evidence).

## How It Works

```text
patient PDF + policy PDF
        ↓
parse once each → OCR when text extraction is insufficient
        ↓
layout / heading detection → logical sections
        ↓
procedure discovery → planned-procedure JSON
        ↓
policy retrieval → criteria tree (or not applicable)
        ↓
evidence verification → MET / MISSING / UNCLEAR per criterion
        ↓
deterministic readiness → READY | NOT_READY | NEEDS_REVIEW | NOT_APPLICABLE
```

## Retrieval Pipeline

### Patient procedure retrieval

```text
patient sections
  → BGE-M3 over headings only
  → select top sections
  → chunk only selected section bodies
  → BM25 + BGE-M3 on selected chunks
  → RRF
  → cross-encoder rerank
  → Luna → planned-procedure JSON
```

RRF is used only in this procedure chunk stage.

### Policy criteria retrieval

```text
procedure JSON
  → exact CPT search + BM25 across all policy sections
  → union / deduplicate
  → BGE-M3 dense on candidates only
  → cross-encoder governing-section rerank
  → local structural expansion around selected sections
  → dedupe + token-budgeted governing context
  → ONE Sol call → matching decision + structured criteria tree
```

There is no full-policy dense retrieval. Matching and criteria extraction share
a single Sol call (`matching_policy_criteria_found` + `criteria_tree`).

### Patient evidence verification

```text
all policy criteria
  → one request-scoped patient BM25 index
  → criterion-specific section candidates
  → deduplicated chunking of unique selected sections once
  → batched BGE-M3 over criteria + unique allowed chunks
  → batched cross-encoder over criterion/chunk pairs
  → token-budgeted batched Sol evaluation
  → MET / MISSING / UNCLEAR per criterion
  → deterministic readiness
```

Evidence reuses the already-parsed patient sections. There is no RRF and no
second BM25 pass over chunks. Criterion isolation is preserved: a criterion
may only search chunks from its own BM25-selected sections.

## Policy Reasoning

Policy Sol returns a structured criteria tree (groups, conditions, exclusions)
grounded in the retrieved governing context. Compact policy citations are
hydrated from allowed candidate text. Payer identity is not matched.

### Temporal evidence

- Explicit encounter/document dates found in patient text are preserved on
  sections and chunks.
- Policy CONDITION nodes may include an optional `temporal_requirement`
  extracted from the same Sol criteria call (for example `current` or
  `within the previous 6 months`).
- Dates are included in dense/rerank text and Sol evidence payloads.
- Historical evidence is not globally discarded; only explicit within-N-months
  windows may drop clearly out-of-window dated evidence. Unknown dates are
  never treated as current.

## Evidence Evaluation

Evidence Sol judges each criterion as `MET` / `MISSING` / `UNCLEAR` using only
that criterion's allowed candidates. Citations are validated against candidate
chunk text and page ranges. Batches shrink dynamically when the configured
evidence context budget would otherwise be exceeded.

## Deterministic Readiness

After evidence evaluation, readiness walks the criteria tree:

- `AND` / `OR` composition of leaf statuses
- exclusions that trigger force `NOT_READY`
- `MET` / `MISSING` / `UNCLEAR` propagation
- final status: `READY`, `NOT_READY`, `NEEDS_REVIEW`, or `NOT_APPLICABLE`

## Setup

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
```

Set at least:

```bash
OPENAI_API_KEY=
```

For scanned PDFs:

```bash
MISTRAL_API_KEY=
```

`HF_TOKEN` is optional (helps with Hugging Face rate limits).

Never commit `.env`. Use `.env.example` as the template only.

## Running

```bash
make run
```

Requires Node.js 20.19+ (or 22.12+) with npm, Python 3.12+, and uv.

This runs `uv sync`, preserves the existing model download/cache preparation,
installs frontend dependencies from the npm lockfile, then launches both services:

- Frontend: http://localhost:5173
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs

**Ctrl+C stops both services and their child processes.** If either service exits,
the launcher stops the other. Ports 8000 and 5173 must be available.

The React/TypeScript/Vite frontend sends relative `/api/analyze` requests. Vite
proxies `/api` and `/health` to `http://127.0.0.1:8000`, without a request timeout
for lengthy policies. Files stay in browser memory only; analysis is never
automatically retried.

The UI faithfully presents `overall_status` (including `NEEDS_REVIEW`), citations,
triggered exclusions, backend-curated `missing_items` / `review_items`, and
`next_step`. The current analysis response does not expose a reference date,
exclusion titles, or an AND/OR criteria tree; the UI does not invent them.
All policy criteria therefore lists the returned evaluated conditions.

Frontend validation:

```bash
cd frontend
npm run build
npx playwright install chromium   # first browser-test run only
npm test
```

Browser tests use mocked API responses, never paid provider calls. For a live
review, choose the patient and policy PDFs in the UI and click Run verification.
A large policy can take several minutes.

For reload during development:

```bash
make dev
```

Models download on first run and are cached afterward (see
`scripts/download_models.py`).

Sample curl against included PDFs:

```bash
curl -X POST \
  -F "patient_file=@sample_data/patient.pdf" \
  -F "policy_file=@sample_data/bcbs_policy.pdf" \
  http://localhost:8000/api/analyze
```

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Configuration readiness |
| `POST` | `/api/analyze` | Analyze patient and policy PDFs |

## Testing

```bash
make check
```

Runs pytest, Ruff, and mypy.

```bash
make test    # pytest only
make lint    # ruff
make typecheck  # mypy app
```

## Configuration

Retrieval knobs map to the three paths:

| Setting | Path |
|---|---|
| `PATIENT_HEADING_TOP_K` | Procedure heading selection |
| `BM25_TOP_K` / `DENSE_TOP_K` / `RRF_TOP_K` / `RRF_K` | Procedure chunk hybrid |
| `PROCEDURE_RERANK_TOP_K` | Procedure rerank |
| `POLICY_BM25_TOP_K` / `POLICY_DENSE_TOP_K` / `POLICY_RERANK_TOP_K` | Policy retrieval |
| `EVIDENCE_SECTION_BM25_TOP_K` / `EVIDENCE_DENSE_TOP_K` / `EVIDENCE_RERANK_TOP_K` | Evidence retrieval |
| `EVIDENCE_SOL_MAX_ITEMS_PER_BATCH` / `EVIDENCE_SOL_MAX_CONCURRENCY` | Evidence Sol batching |
| `EMBEDDING_BATCH_SIZE` | Local embedding/reranker batch size |
| `SECTION_NEIGHBOR_WINDOW` | Policy context expansion |
| `COMPACT_SECTION_CHARS` | Section BM25 preview width |

Observability defaults (quiet normal runs):

| Setting | Default | Purpose |
|---|---|---|
| `DEBUG_TRACE_ENABLED` | `false` | Write detailed JSON traces under `debug_trace/` |
| `PERFORMANCE_PROFILE_ENABLED` | `false` | Emit detailed substage / pair-duplication reports |

See `.env.example` for the full list.

Normal `/api/analyze` logs emit one structured `analysis_request_summary` with
request id, status, stage timings, page/criteria counts, and model usage.
Detailed sensitive document text belongs only in explicit debug mode.

## Assumptions

- Uploaded files are PDFs (patient chart + medical-necessity policy).
- The uploaded policy is treated as the governing document; payer identity is
  not matched.
- Procedure discovery targets the next planned procedure in the chart.
- Final readiness is deterministic given the criteria tree and evidence
  statuses.

## Failure Modes / Limitations

- Missing credentials fail fast with configuration errors.
- Unreadable pages after OCR handling raise processing errors.
- LLM outputs are validated against cited chunk text and page ranges.
- Empty retrieval candidates terminate that path without inventing results.
- Policy and evidence prompts use explicit tiktoken context budgets and
  bounded `max_completion_tokens` output limits.
- OpenAI calls are independent (no accumulating chat history across stages).
- Retries are application-owned only (`LLM_MAX_RETRIES`, SDK `max_retries=0`)
  with bounded exponential backoff + jitter, and only for transient failures
  (timeouts, connection errors, 429, 5xx).
- Large multi-thousand-page policies are slow on CPU (see Performance Notes).
- OCR quality limits recovery for poorly scanned layouts.
- Frontend UI is not part of this backend MVP yet.

## Performance Notes

Cold processing of a very large ~3,655-page policy is dominated by CPU
cross-encoder evidence reranking and policy structured-output generation.

Future production optimization (not implemented in this MVP):

- pre-index/cache stable policies by hash
- accelerator/batched reranking
- retrieval regression testing before reducing candidate breadth

## AI Tool Usage

_(To be filled for final submission: tools used, what they helped with, and
what was verified manually.)_
