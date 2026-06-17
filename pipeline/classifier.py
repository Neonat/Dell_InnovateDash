from scipy.special import softmax
from sentence_transformers import CrossEncoder

from pipeline.models import ClassificationResult

_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

PROBLEM_BUCKETS = [
    "financial_assistance",
    "mobility_support",
    "medical_care",
    "mental_health",
    "caregiving_support",
    "social_isolation",
    "housing",
    "legal_and_admin",
    "end_of_life",
]

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

_THRESHOLD = 0.15


def classify(text: str) -> ClassificationResult:
    pairs = [
        (text, f"This person needs help with {BUCKET_DESCRIPTIONS[b]}.")
        for b in PROBLEM_BUCKETS
    ]
    scores = _nli_model.predict(pairs)

    matched = []
    best_score = 0.0
    for bucket, score_arr in zip(PROBLEM_BUCKETS, scores):
        # score_arr is raw logits [contradiction, entailment, neutral]
        probs = softmax(score_arr)
        entailment = float(probs[1])
        if entailment > _THRESHOLD:
            matched.append(bucket)
        if entailment > best_score:
            best_score = entailment

    if matched:
        return ClassificationResult(
            problem_classes=matched,
            special_case=None,
            confidence=best_score,
        )

    # No bucket matched — extract a one-sentence description
    sentences = [s.strip() for s in text.replace(".", ". ").split(". ") if s.strip()]
    special = sentences[0] if sentences else text[:200]
    return ClassificationResult(
        problem_classes=[],
        special_case=special,
        confidence=best_score,
    )
