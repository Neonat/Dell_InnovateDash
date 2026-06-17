from pipeline.sentiment import analyse


def test_positive_sentiment():
    result = analyse("I feel much better after getting the grant")
    assert result.label == "positive"
    assert result.escalated is False


def test_high_escalation():
    result = analyse("I feel like there is no point living anymore")
    assert result.escalated is True
    assert result.escalation_severity == "high"


def test_neutral_no_escalation():
    result = analyse("I need help with my medication costs")
    assert result.escalated is False
    assert result.label in ("neutral", "negative")


def test_high_escalation_emotion_override():
    result = analyse("I want to die, nobody cares about me")
    assert result.emotion == "distressed"
    assert result.escalation_severity == "high"
