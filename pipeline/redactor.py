from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from pipeline.models import PiiEntity, RedactedText

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

_nric_recognizer = PatternRecognizer(
    supported_entity="NRIC",
    patterns=[Pattern(name="nric", regex=r"\b[STFG]\d{7}[A-Z]\b", score=0.95)],
)
_analyzer.registry.add_recognizer(_nric_recognizer)

_ENTITIES = [
    "PERSON",
    "NRIC",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "DATE_OF_BIRTH",
    "CREDIT_CARD",
]


def redact(text: str) -> RedactedText:
    results = _analyzer.analyze(text=text, entities=_ENTITIES, language="en")

    anonymized = _anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<{entity_type}>"})},
    )

    entities = [
        PiiEntity(
            entity_type=r.entity_type,
            original=text[r.start:r.end],
            placeholder=f"<{r.entity_type}>",
            start=r.start,
            end=r.end,
        )
        for r in results
    ]

    return RedactedText(
        original=text,
        redacted=anonymized.text,
        entities=entities,
    )
