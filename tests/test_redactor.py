from pipeline.redactor import redact


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


def test_original_preserved():
    text = "Call me at 91234567"
    result = redact(text)
    assert result.original == text
