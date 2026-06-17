from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PiiEntity:
    entity_type: str
    original: str
    placeholder: str
    start: int
    end: int


@dataclass
class RedactedText:
    original: str
    redacted: str
    entities: list[PiiEntity]


@dataclass
class SentimentResult:
    label: str
    score: float
    emotion: str
    escalated: bool
    escalation_severity: Optional[str]
    escalation_reason: Optional[str]


@dataclass
class ClassificationResult:
    problem_classes: list[str]
    special_case: Optional[str]
    confidence: float


@dataclass
class PipelineOutput:
    text: str
    sentiment: dict
    escalated: bool
    escalation_severity: Optional[str]
    problem_classes: list[str]
    special_case: Optional[str]
