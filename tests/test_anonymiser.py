from pipeline.redactor import redact
from pipeline.anonymiser import anonymise, restore
from pipeline.models import RedactedText, PiiEntity


def test_placeholder_format():
    redacted = RedactedText(
        original="David Tan, S1234567A",
        redacted="<PERSON>, <NRIC>",
        entities=[
            PiiEntity(entity_type="PERSON", original="David Tan", placeholder="<PERSON>", start=0, end=9),
            PiiEntity(entity_type="NRIC", original="S1234567A", placeholder="<NRIC>", start=11, end=20),
        ],
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


def test_multiple_same_type():
    redacted = RedactedText(
        original="Alice and Bob",
        redacted="<PERSON> and <PERSON>",
        entities=[
            PiiEntity(entity_type="PERSON", original="Alice", placeholder="<PERSON>", start=0, end=5),
            PiiEntity(entity_type="PERSON", original="Bob", placeholder="<PERSON>", start=10, end=13),
        ],
    )
    anon, mapping = anonymise(redacted)
    assert "[PERSON_1]" in anon
    assert "[PERSON_2]" in anon
    assert mapping["[PERSON_1]"] == "Alice"
    assert mapping["[PERSON_2]"] == "Bob"
