import argparse
import json

from pipeline.redactor import redact
from pipeline.anonymiser import anonymise, restore
from pipeline.sentiment import analyse, HOLDING_MESSAGE
from pipeline.llm import call_llm
from pipeline.classifier import classify
from pipeline.output import assemble

MEDIUM_APPENDAGE = "\n\nIf things ever feel overwhelming, you can also reach out to 1800-650-6060."


def process(text: str) -> dict:
    redacted = redact(text)
    anon_text, mapping = anonymise(redacted)
    sentiment = analyse(text)

    if sentiment.escalation_severity == "high":
        classification = classify(text)
        output = assemble(
            llm_text=HOLDING_MESSAGE,
            sentiment=sentiment,
            classification=classification,
        )
        return output.__dict__

    llm_response_anon = call_llm(anon_text)
    llm_response = restore(llm_response_anon, mapping)

    if sentiment.escalation_severity == "medium":
        llm_response += MEDIUM_APPENDAGE

    classification = classify(text)
    output = assemble(llm_response, sentiment, classification)
    return output.__dict__


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    result = process(args.text)
    print(json.dumps(result, indent=2))
