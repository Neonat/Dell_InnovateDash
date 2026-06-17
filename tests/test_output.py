from pipeline.output import assemble
from pipeline.models import SentimentResult, ClassificationResult


def _make_sentiment(**kwargs):
    defaults = dict(
        label="negative",
        score=0.781,
        emotion="anxious",
        escalated=False,
        escalation_severity=None,
        escalation_reason=None,
    )
    defaults.update(kwargs)
    return SentimentResult(**defaults)


def _make_classification(**kwargs):
    defaults = dict(problem_classes=["medical_care"], special_case=None, confidence=0.8)
    defaults.update(kwargs)
    return ClassificationResult(**defaults)


def test_normal_output_shape():
    out = assemble("Some response text", _make_sentiment(), _make_classification())
    assert out.text == "Some response text"
    assert out.escalated is False
    assert out.escalation_severity is None
    assert "label" in out.sentiment
    assert "score" in out.sentiment
    assert "emotion" in out.sentiment


def test_escalated_output():
    sentiment = _make_sentiment(
        label="negative",
        score=0.967,
        emotion="distressed",
        escalated=True,
        escalation_severity="high",
    )
    out = assemble("Holding message here.", sentiment, _make_classification(problem_classes=["mental_health"]))
    assert out.escalated is True
    assert out.escalation_severity == "high"
    assert out.problem_classes == ["mental_health"]


def test_score_rounded():
    out = assemble("text", _make_sentiment(score=0.78123456), _make_classification())
    assert out.sentiment["score"] == 0.781


def test_special_case_output():
    classification = _make_classification(problem_classes=[], special_case="Unusual neighbour dispute.", confidence=0.2)
    out = assemble("response", _make_sentiment(), classification)
    assert out.problem_classes == []
    assert out.special_case == "Unusual neighbour dispute."
