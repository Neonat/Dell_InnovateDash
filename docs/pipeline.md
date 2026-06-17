# Pipeline spec

## Overview

Input: plain text string (speech already converted upstream)
Output: `PipelineOutput` JSON

```
text input
    │
    ▼
Stage 1: redactor.py       — detect PII, return entity map + redacted text
    │
    ▼
Stage 2: anonymiser.py     — replace PII with typed placeholders
    │
    ▼
Stage 3: sentiment.py      — sentiment label + score + escalation check
    │
    ├── if escalated ──────→ skip LLM, go straight to output.py with holding text
    │
    ▼
Stage 4: llm.py            — call external LLM with anonymised text
    │
    ▼
Stage 2 (restore):         — swap placeholders back to real values in LLM response
    │
    ▼
Stage 5: classifier.py     — classify into social service problem buckets
    │
    ▼
Stage 6: output.py         — assemble and return PipelineOutput JSON
```

---

## Shared dataclasses (models.py)

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PiiEntity:
    entity_type: str      # e.g. "PERSON", "NRIC", "ADDRESS", "PHONE_NUMBER"
    original: str         # the actual value found
    placeholder: str      # e.g. "[PERSON_1]"
    start: int
    end: int

@dataclass
class RedactedText:
    original: str
    redacted: str
    entities: list[PiiEntity]

@dataclass
class SentimentResult:
    label: str            # "positive" | "neutral" | "negative"
    score: float          # confidence 0.0–1.0
    emotion: str          # fine-grained: "hopeful" | "distressed" | "neutral" | "frustrated" | "grateful" | "anxious"
    escalated: bool
    escalation_severity: Optional[str]   # None | "low" | "medium" | "high"
    escalation_reason: Optional[str]     # human-readable trigger description

@dataclass
class ClassificationResult:
    problem_classes: list[str]   # one or more from the standard bucket list
    special_case: Optional[str]  # free-text description if no bucket fits, else None
    confidence: float

@dataclass
class PipelineOutput:
    text: str
    sentiment: dict                      # serialised SentimentResult (minus escalation_reason)
    escalated: bool
    escalation_severity: Optional[str]
    problem_classes: list[str]
    special_case: Optional[str]
```

---

## Stage 1 — redactor.py

**Purpose**: detect PII in the raw input text using Microsoft Presidio running fully locally.

**Input**: `str`
**Output**: `RedactedText`

### Implementation

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
```

**Entities to detect** (configure Presidio to look for these):
- `PERSON` — full names
- `NRIC` — custom recogniser for Singapore NRIC format: `[STFG]\d{7}[A-Z]`
- `PHONE_NUMBER`
- `EMAIL_ADDRESS`
- `LOCATION` — addresses, block numbers, estate names
- `DATE_OF_BIRTH`
- `CREDIT_CARD`

**Custom Presidio recogniser for NRIC** — add to `redactor.py`:

```python
from presidio_analyzer import PatternRecognizer, Pattern

nric_recognizer = PatternRecognizer(
    supported_entity="NRIC",
    patterns=[Pattern(name="nric", regex=r"\b[STFG]\d{7}[A-Z]\b", score=0.95)]
)
analyzer.registry.add_recognizer(nric_recognizer)
```

**Output format**: Return `RedactedText` with `redacted` field containing the original text with PII replaced by entity type tokens (`<PERSON>`, `<NRIC>`, etc.) — the anonymiser in stage 2 will rename these to numbered placeholders.

### Test cases

```python
# test_redactor.py
def test_nric_detected():
    result = redact("My NRIC is S1234567A")
    assert any(e.entity_type == "NRIC" for e in result.entities)

def test_name_detected():
    result = redact("My name is David Tan")
    assert any(e.entity_type == "PERSON" for e in result.entities)

def test_no_pii_unchanged():
    result = redact("I cannot afford my medication")
    assert result.entities == []
    assert result.redacted == "I cannot afford my medication"
```

---

## Stage 2 — anonymiser.py

**Purpose**: replace detected PII with stable, typed, numbered placeholders before the LLM call. Restore them after.

**Input**: `RedactedText`
**Output**: `str` (anonymised text with placeholders) + a `restore()` function

### Placeholder format

`[ENTITY_TYPE_N]` where N increments per entity type per request.

Example: two different people in one text → `[PERSON_1]` and `[PERSON_2]`.

### Implementation

```python
def anonymise(redacted: RedactedText) -> tuple[str, dict[str, str]]:
    """
    Returns:
        anonymised_text: str with [ENTITY_N] placeholders
        mapping: dict mapping placeholder → original value, for restoration
    """

def restore(text: str, mapping: dict[str, str]) -> str:
    """Replace all [ENTITY_N] placeholders in text with original values."""
```

The `mapping` dict is kept in memory for the duration of the request only — never persisted.

### Test cases

```python
def test_placeholder_format():
    redacted = RedactedText(
        original="David Tan, S1234567A",
        redacted="<PERSON>, <NRIC>",
        entities=[...]
    )
    anon, mapping = anonymise(redacted)
    assert "[PERSON_1]" in anon
    assert "[NRIC_1]" in anon
    assert mapping["[PERSON_1]"] == "David Tan"

def test_restore_roundtrip():
    original = "David Tan called about his NRIC S1234567A"
    redacted = redact(original)
    anon, mapping = anonymise(redacted)
    llm_response = f"I can help [PERSON_1] with their query about [NRIC_1]."
    restored = restore(llm_response, mapping)
    assert "David Tan" in restored
    assert "S1234567A" in restored
```

---

## Stage 3 — sentiment.py

**Purpose**: classify the emotional tone of the input. Detect distress signals for escalation.

**Input**: `str` (original text, before anonymisation — sentiment needs real language)
**Output**: `SentimentResult`

### Sentiment model

Primary: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Labels: `positive`, `neutral`, `negative`
- Load once at module import, not per request

### Emotion mapping

After getting the primary label + score, run a second pass to assign a fine-grained emotion:

```python
EMOTION_MAP = {
    # keyword → emotion label
    "hopeless": "distressed",
    "no point": "distressed",
    "alone": "distressed",
    "scared": "anxious",
    "worried": "anxious",
    "angry": "frustrated",
    "unfair": "frustrated",
    "thank": "grateful",
    "hope": "hopeful",
    "better": "hopeful",
}
# default to "neutral" if no keyword matches
```

### Escalation tiers

Run keyword scan first (fast), then NLI for ambiguous cases.

**Tier 1 — high severity** (keyword match → escalate immediately, skip NLI):

```python
TIER1 = [
    "want to die", "no point living", "end my life", "kill myself",
    "rather be dead", "don't want to live", "suicide",
    "hits me", "hurts me", "beats me", "afraid to go home",
    "chest pain", "can't breathe", "collapsed",
]
```

**Tier 2 — medium severity** (NLI check if no Tier 1 match):

Model: `cross-encoder/nli-deberta-v3-small`, runs locally.
Hypothesis: `"The person is expressing distress, fear, or danger."`
Threshold: entailment score > 0.65 → `escalation_severity = "medium"`

**On escalation**, `text` in the final output is replaced with a holding message:

```
"It sounds like you're going through something difficult. 
Please reach out to the helpline at 1800-650-6060, or SOS at 1767 if urgent.
A care advisor will follow up with you shortly."
```

The LLM is NOT called when `escalation_severity == "high"`. It IS still called for `"medium"` but the holding message is appended to the response.

### Test cases

```python
def test_positive_sentiment():
    result = analyse("I feel much better after getting the grant")
    assert result.label == "positive"
    assert result.escalated == False

def test_high_escalation():
    result = analyse("I feel like there is no point living anymore")
    assert result.escalated == True
    assert result.escalation_severity == "high"

def test_neutral_no_escalation():
    result = analyse("I need help with my medication costs")
    assert result.escalated == False
    assert result.label in ("neutral", "negative")
```

---

## Stage 4 — classifier.py

**Purpose**: classify the input into one or more standard social service problem buckets. If nothing fits, create a special case.

**Input**: `str` (original text)
**Output**: `ClassificationResult`

### Standard buckets

```python
PROBLEM_BUCKETS = [
    "financial_assistance",    # bills, grants, subsidies, income support
    "mobility_support",        # assistive devices, home modification, transport
    "medical_care",            # medication, hospital, chronic conditions
    "mental_health",           # depression, anxiety, loneliness, grief
    "caregiving_support",      # caregiver burnout, FDW, respite care
    "social_isolation",        # befriending, activity centres, community
    "housing",                 # rental, flat upgrade, maintenance
    "legal_and_admin",         # LPA, advance care planning, benefits paperwork
    "end_of_life",             # palliative care, hospice, funeral assistance
]
```

### Implementation

Use zero-shot classification via NLI. For each bucket, run:

```
Hypothesis: "This person needs help with {bucket_description}."
```

Score all buckets. Return any with entailment score > 0.5. If none exceed 0.5, return `special_case` with a one-sentence free-text description of the need extracted from the input.

Model: `cross-encoder/nli-deberta-v3-small` (same model as escalation — load once, reuse).

```python
BUCKET_DESCRIPTIONS = {
    "financial_assistance": "financial assistance, grants, subsidies, or income support",
    "mobility_support": "mobility aids, assistive devices, or home modifications",
    "medical_care": "medical care, medication costs, or chronic illness management",
    "mental_health": "mental health, emotional wellbeing, loneliness, or grief",
    "caregiving_support": "caregiving, caregiver support, or respite care",
    "social_isolation": "social connection, befriending, or community activities",
    "housing": "housing, rental assistance, or home maintenance",
    "legal_and_admin": "legal matters, advance care planning, or administrative help",
    "end_of_life": "palliative care, hospice, or end-of-life planning",
}
```

### Test cases

```python
def test_financial_bucket():
    result = classify("I cannot afford my medication and utilities")
    assert "financial_assistance" in result.problem_classes
    assert "medical_care" in result.problem_classes

def test_special_case():
    result = classify("My neighbour keeps stealing my mail and I don't know who to call")
    assert result.problem_classes == []
    assert result.special_case is not None
    assert len(result.special_case) > 10

def test_multiple_buckets():
    result = classify("I am lonely and also need help with my FDW paperwork")
    assert "social_isolation" in result.problem_classes
    assert "caregiving_support" in result.problem_classes
```

---

## Stage 5 — output.py

**Purpose**: assemble the final `PipelineOutput` and serialise to JSON.

**Input**: `SentimentResult`, `ClassificationResult`, LLM response text (with PII restored)
**Output**: `PipelineOutput` as a JSON-serialisable dict

### Assembly logic

```python
def assemble(
    llm_text: str,
    sentiment: SentimentResult,
    classification: ClassificationResult,
) -> PipelineOutput:
    return PipelineOutput(
        text=llm_text,
        sentiment={
            "label": sentiment.label,
            "score": round(sentiment.score, 3),
            "emotion": sentiment.emotion,
        },
        escalated=sentiment.escalated,
        escalation_severity=sentiment.escalation_severity,
        problem_classes=classification.problem_classes,
        special_case=classification.special_case,
    )
```

### Final JSON examples

**Normal response:**
```json
{
  "text": "It sounds like you may benefit from the Home Caregiving Grant and mobility support schemes. Can you tell me more about your living situation?",
  "sentiment": {
    "label": "negative",
    "score": 0.781,
    "emotion": "anxious"
  },
  "escalated": false,
  "escalation_severity": null,
  "problem_classes": ["caregiving_support", "mobility_support"],
  "special_case": null
}
```

**Escalated response:**
```json
{
  "text": "It sounds like you're going through something difficult. Please reach out to the helpline at 1800-650-6060, or SOS at 1767 if urgent. A care advisor will follow up with you shortly.",
  "sentiment": {
    "label": "negative",
    "score": 0.967,
    "emotion": "distressed"
  },
  "escalated": true,
  "escalation_severity": "high",
  "problem_classes": ["mental_health"],
  "special_case": null
}
```

**Special case:**
```json
{
  "text": "That sounds like a unique situation. Let me connect you with someone who can help.",
  "sentiment": {
    "label": "neutral",
    "score": 0.612,
    "emotion": "neutral"
  },
  "escalated": false,
  "escalation_severity": null,
  "problem_classes": [],
  "special_case": "Person is experiencing neighbour harassment and does not know which agency to contact."
}
```

---

## run.py — wiring it all together

```python
# pipeline/run.py
from pipeline.redactor import redact
from pipeline.anonymiser import anonymise, restore
from pipeline.sentiment import analyse
from pipeline.llm import call_llm
from pipeline.classifier import classify
from pipeline.output import assemble

def process(text: str) -> dict:
    # Stage 1: redact
    redacted = redact(text)

    # Stage 2: anonymise
    anon_text, mapping = anonymise(redacted)

    # Stage 3: sentiment + escalation (on original text)
    sentiment = analyse(text)

    if sentiment.escalation_severity == "high":
        # skip LLM, return holding message immediately
        classification = classify(text)
        output = assemble(
            llm_text=HOLDING_MESSAGE,
            sentiment=sentiment,
            classification=classification,
        )
        return output.__dict__

    # Stage 4: LLM call (anonymised input)
    llm_response_anon = call_llm(anon_text)

    # Stage 2 (restore): put PII back into LLM response
    llm_response = restore(llm_response_anon, mapping)

    # Stage 5: classify
    classification = classify(text)

    # Stage 6: assemble output
    output = assemble(llm_response, sentiment, classification)
    return output.__dict__
```

---

## Demo checkpoint

```bash
# Normal input
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is Mdm Lim, I am 74. I cannot afford my dialysis trips."}'

# Expected: PII stripped before LLM, restored in response, buckets include medical_care + financial_assistance

# Escalation input
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel like there is no point living anymore. Nobody cares."}'

# Expected: escalated=true, severity=high, holding message returned, LLM not called

# Special case input
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "My town council refuses to fix the ceiling and I have been waiting 8 months"}'

# Expected: problem_classes=[], special_case contains a description of the issue
```
