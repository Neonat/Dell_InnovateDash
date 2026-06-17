import re
from scipy.special import softmax
from transformers import pipeline as hf_pipeline
from sentence_transformers import CrossEncoder

from pipeline.models import SentimentResult

_sentiment_pipe = hf_pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=1,
)

_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

TIER1_KEYWORDS = [
    "want to die",
    "no point living",
    "end my life",
    "kill myself",
    "rather be dead",
    "don't want to live",
    "suicide",
    "take my own life",
    "hits me",
    "hurts me",
    "beats me",
    "afraid to go home",
    "lock me in",
    "take my money",
    "threatening me",
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "collapsed",
    "unconscious",
]

EMOTION_MAP = {
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

HOLDING_MESSAGE = (
    "It sounds like you're going through something difficult. \n"
    "Please reach out to the helpline at 1800-650-6060, or SOS at 1767 if urgent.\n"
    "A care advisor will follow up with you shortly."
)

_LABEL_MAP = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}


def _normalise_label(raw: str) -> str:
    return _LABEL_MAP.get(raw.lower(), raw.lower())


def _detect_emotion(text: str) -> str:
    lower = text.lower()
    for keyword, emotion in EMOTION_MAP.items():
        if keyword in lower:
            return emotion
    return "neutral"


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def analyse(text: str) -> SentimentResult:
    stripped = _strip_punctuation(text)

    # Tier 1 keyword check
    for keyword in TIER1_KEYWORDS:
        if keyword in stripped:
            raw = _sentiment_pipe(text)
            while isinstance(raw, list):
                raw = raw[0]
            label = _normalise_label(raw["label"])
            score = float(raw["score"])
            return SentimentResult(
                label=label,
                score=score,
                emotion="distressed",
                escalated=True,
                escalation_severity="high",
                escalation_reason=f"Tier 1 keyword matched: '{keyword}'",
            )

    # Sentiment
    raw = _sentiment_pipe(text)
    while isinstance(raw, list):
        raw = raw[0]
    label = _normalise_label(raw["label"])
    score = float(raw["score"])
    emotion = _detect_emotion(text)

    # Tier 2 NLI check
    scores = _nli_model.predict(
        [(text, "The person is expressing distress, fear, or danger.")]
    )
    # CrossEncoder NLI returns raw logits [contradiction, entailment, neutral]
    probs = softmax(scores[0])
    entailment_score = float(probs[1])

    if entailment_score > 0.97:
        return SentimentResult(
            label=label,
            score=score,
            emotion=emotion,
            escalated=True,
            escalation_severity="medium",
            escalation_reason=f"NLI entailment score {entailment_score:.3f} exceeded threshold",
        )

    return SentimentResult(
        label=label,
        score=score,
        emotion=emotion,
        escalated=False,
        escalation_severity=None,
        escalation_reason=None,
    )
