# Social-service text pipeline

A Python prototype built for the **Dell InnovateDash 2026 hackathon** to turn a social-service conversation into a structured response and triage signals. It combines local PII detection, anonymisation, sentiment and escalation signals, response generation, and problem-category classification.

The implemented service accepts plain text through a FastAPI endpoint or command-line entry point and returns JSON. It has no frontend, speech-to-text component or database.

## Processing flow

1. Detect supported personal-data entities locally with Presidio and create numbered placeholders.
2. Analyse the original text with local sentiment and natural-language-inference models.
3. For a high-severity signal, skip the external LLM request and return the configured holding response with classification results.
4. For other inputs, send the anonymised text to the configured OpenAI-compatible provider, restore recognised placeholders in the response, and append the configured support text for medium-severity signals.
5. Classify the input into social-service categories and assemble the JSON result.

The local models are `cardiffnlp/twitter-roberta-base-sentiment-latest`, `cross-encoder/nli-deberta-v3-small`, and spaCy's `en_core_web_lg`. Docker installs Python 3.11 and the dependencies listed in [requirements.txt](requirements.txt).

## Run with Docker Compose

```bash
git clone https://github.com/Neonat/social-service-text-pipeline.git
cd social-service-text-pipeline
cp .env.example .env
```

Set `LLM_API_KEY` in `.env`. Check `LLM_BASE_URL` and `MODEL` against the provider and model you intend to use; the values in the template describe the original development setup.

```bash
docker compose up --build
```

The initial build downloads Python dependencies and the local language models. Compose exposes the API on **host port 8005**, mapped to port 8000 in the container.

```bash
curl -X POST http://localhost:8005/process \
  -H 'Content-Type: application/json' \
  -d '{"text":"I need help understanding the available community services."}'
```

The LLM client is initialised during startup, so a configured key is required even when a particular high-severity request would skip the external call.

## Command-line use

With the Compose service running:

```bash
docker compose exec app python -m pipeline.run \
  --text "I need help understanding the available community services."
```

## Result fields

| Field | Meaning |
| --- | --- |
| `text` | Generated response, or the configured high-severity holding response |
| `sentiment` | Sentiment label, score and keyword-derived emotion |
| `escalated` | Whether an escalation signal was detected |
| `escalation_severity` | `high`, `medium`, or `null` |
| `problem_classes` | Matched social-service categories |
| `special_case` | A short extract when no category matches |

## Repository guide

- [pipeline/run.py](pipeline/run.py): processing order and CLI.
- [pipeline/server.py](pipeline/server.py): `POST /process` endpoint.
- [pipeline/redactor.py](pipeline/redactor.py) and [pipeline/anonymiser.py](pipeline/anonymiser.py): entity handling and placeholder restoration.
- [pipeline/sentiment.py](pipeline/sentiment.py): sentiment, keywords and escalation thresholds.
- [pipeline/classifier.py](pipeline/classifier.py): category descriptions and classification threshold.
- [tests](tests/): component tests.
- [docs/pipeline.md](docs/pipeline.md) and [docs/escalation.md](docs/escalation.md): original design notes and stage-level details.

## Checks

In the configured container environment:

```bash
docker compose exec app pytest -v
```

Some tests import the local language models and require their dependencies and downloads. A small output-assembly check can be run separately with `pytest tests/test_output.py -v` in a Python environment with pytest installed.

## Prototype boundaries

Entity detection is model- and recogniser-based and can miss sensitive content. Restored response text and the `special_case` field may contain original input. The prototype has no implemented human follow-up system, despite the wording of its configured holding response, and it has not been established as a clinical or emergency-response service.

The original design notes contain historical thresholds and examples. Current behaviour is defined by the source files linked above: the medium escalation threshold is `0.97`, the classification threshold is `0.15`, and high-severity inputs bypass response generation. Publication of the repository does not add an open-source licence; none is currently provided.
