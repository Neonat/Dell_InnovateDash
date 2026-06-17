# Responsible AI pipeline

A self-contained Python pipeline that takes text input (assumed already converted from speech) and runs it through five sequential stages: PII redaction, anonymous LLM inference, sentiment analysis, problem classification, and structured JSON output for a downstream 2D model renderer.

No frontend. No product name references. No form filling. No vector DB. Input is plain text, output is JSON.

## Tech stack

- **Runtime**: Python 3.11, FastAPI, uvicorn
- **LLM**: Groq API (llama3-8b-8192) via `pipeline/llm.py` — OpenAI-compatible, swap via env
- **PII detection**: `presidio-analyzer` + `presidio-anonymizer` (runs fully locally)
- **Sentiment / NLI**: `cardiffnlp/twitter-roberta-base-sentiment-latest` + `cross-encoder/nli-deberta-v3-small` (local, HuggingFace)
- **Infra**: Docker Compose only — single `app` container, no database needed for this stage

## Commands

```bash
# start
docker compose up --build

# run the full pipeline against a test input
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is David Tan, NRIC S1234567A. I live at Blk 45 Bedok North. I feel hopeless and alone."}'

# run tests
docker compose exec app pytest -v

# run pipeline directly (no server)
docker compose exec app python -m pipeline.run --text "I am 70 and cannot afford my medication"
```

## Project layout

```
project/
├── CLAUDE.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── __init__.py
│   ├── run.py            # CLI entry point, runs full pipeline end to end
│   ├── redactor.py       # Stage 1: PII detection and redaction using Presidio
│   ├── anonymiser.py     # Stage 2: replace PII tokens with placeholders, restore after LLM
│   ├── llm.py            # LLM client — anonymised text in, response out
│   ├── sentiment.py      # Stage 3: sentiment analysis → label + score
│   ├── classifier.py     # Stage 4: problem classification into social service buckets
│   ├── output.py         # Stage 5: assemble final JSON output
│   └── models.py         # shared dataclasses: PipelineInput, RedactedText, PipelineOutput
├── tests/
│   ├── test_redactor.py
│   ├── test_anonymiser.py
│   ├── test_sentiment.py
│   ├── test_classifier.py
│   └── test_output.py
└── docs/
    ├── pipeline.md       # stage-by-stage spec with inputs, outputs, and test cases
    └── escalation.md     # escalation thresholds, severity levels, trigger rules
```

## Key conventions

- **Stages are independent modules.** Each takes a well-typed input and returns a well-typed output. No stage imports from another — they are wired together only in `run.py`.
- **PII never leaves the machine.** Presidio runs locally. The LLM receives placeholder tokens (`[PERSON_1]`, `[NRIC_1]`, `[ADDRESS_1]`), not real values. Restoration happens after the LLM responds.
- **Escalation is synchronous and non-optional.** `sentiment.py` returns an `EscalationFlag` if distress is detected. `run.py` checks this before calling the LLM. If escalated, the LLM is still called but the output JSON includes `"escalated": true` and `"escalation_severity": "high"`.
- **Every response produces a JSON output**, including escalated ones. The downstream renderer always gets a complete object.
- **No product names, no organisation names** anywhere in code, comments, or output strings.

## Environment variables

Copy `.env.example` to `.env`:

```
LLM_API_KEY=
LLM_BASE_URL=https://api.groq.com/openai/v1
MODEL=llama3-8b-8192
```

That is all. No database URL, no Redis.

## Pipeline stages (summary)

See `docs/pipeline.md` for full spec.

1. **`redactor.py`** — detect PII entities with Presidio, return entity map and redacted text
2. **`anonymiser.py`** — replace detected entities with typed placeholders; restore after LLM call
3. **`llm.py`** — call LLM with anonymised text; response may contain placeholders which are then restored
4. **`sentiment.py`** — run local sentiment model; detect escalation triggers; return label, score, escalation flag
5. **`classifier.py`** — classify the input into one or more social service buckets; create special-case bucket if no match
6. **`output.py`** — assemble `PipelineOutput` dataclass and serialise to JSON

## Final output shape

Every call to `POST /process` returns:

```json
{
  "text": "string — the LLM response with PII restored",
  "sentiment": {
    "label": "positive | neutral | negative",
    "score": 0.0,
    "emotion": "string — e.g. hopeful, distressed, neutral, frustrated"
  },
  "escalated": false,
  "escalation_severity": null,
  "problem_classes": ["financial_assistance", "mobility_support"],
  "special_case": null
}
```

## What NOT to do

- Do not send raw PII to any external API — always pass through `anonymiser.py` first
- Do not skip escalation check — it runs before the LLM call, not after
- Do not add langchain or llamaindex — keep LLM calls direct and inspectable
- Do not return plain strings from any stage — always use the typed dataclasses from `models.py`
- Do not hardcode the LLM provider — read from env
