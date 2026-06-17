# Escalation spec

## Principle

Escalation is checked in `sentiment.py` before the LLM is called. It is synchronous and non-optional. The output JSON always reflects the escalation state so the downstream renderer can act on it.

## Severity levels

| Severity | Trigger | LLM called? | Response text |
|----------|---------|-------------|---------------|
| `high`   | Tier 1 keyword match | No | Holding message + hotlines |
| `medium` | NLI entailment > 0.65 | Yes | LLM response + appended check-in line |
| `null`   | No trigger | Yes | Normal LLM response |

## Tier 1 keyword list

Exact substring match (lowercase, strip punctuation before matching):

```python
TIER1_KEYWORDS = [
    # suicidal ideation
    "want to die",
    "no point living",
    "end my life",
    "kill myself",
    "rather be dead",
    "don't want to live",
    "suicide",
    "take my own life",
    # abuse / safety
    "hits me",
    "hurts me",
    "beats me",
    "afraid to go home",
    "lock me in",
    "take my money",
    "threatening me",
    # acute medical emergency
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "collapsed",
    "unconscious",
]
```

## Tier 2 NLI check

Only runs if no Tier 1 match.

- Model: `cross-encoder/nli-deberta-v3-small`
- Hypothesis: `"The person is expressing distress, fear, or danger."`
- Threshold: entailment score > 0.65

## Holding message (used for high severity)

```
It sounds like you're going through something difficult. 
Please reach out to the helpline at 1800-650-6060, or SOS at 1767 if urgent.
A care advisor will follow up with you shortly.
```

Do not modify this string. It is the exact text placed in `output.text` for high-severity escalations.

## Medium severity appendage

Appended to the end of the normal LLM response:

```
\n\nIf things ever feel overwhelming, you can also reach out to 1800-650-6060.
```

## What the JSON renderer receives on escalation

The `sentiment.emotion` field will always be `"distressed"` on high-severity escalation, regardless of the model's primary sentiment score. This is hardcoded in `sentiment.py` to ensure the renderer shows the correct emotional state.

```json
{
  "text": "...",
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

The renderer should treat `escalated: true` as a signal to display a calm, supportive expression — not a distressed one — since the goal is to de-escalate.
