from pipeline.models import SentimentResult, ClassificationResult, PipelineOutput


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
