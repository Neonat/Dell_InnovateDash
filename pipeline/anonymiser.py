import re
from collections import defaultdict

from pipeline.models import RedactedText


def anonymise(redacted: RedactedText) -> tuple[str, dict[str, str]]:
    counters: dict[str, int] = defaultdict(int)
    mapping: dict[str, str] = {}

    text = redacted.redacted

    # Sort entities by position descending so replacements don't shift offsets
    sorted_entities = sorted(redacted.entities, key=lambda e: e.start, reverse=True)

    # Build placeholder → original mapping, then do a single-pass replace on the
    # Presidio-anonymised text (which already has <ENTITY_TYPE> tokens in it).
    # We process in order of appearance so the same entity type gets incrementing N.
    entity_order = sorted(redacted.entities, key=lambda e: e.start)
    placeholder_map: dict[str, str] = {}  # "<TYPE>" occurrence index → "[TYPE_N]"

    # Map each unique Presidio token occurrence to a numbered placeholder
    numbered: dict[str, str] = {}  # maps "<TYPE>" → latest "[TYPE_N]" (for simple 1-per-type case)
    # We need per-occurrence numbering, so iterate in order
    occurrence_mapping: list[tuple[str, str, str]] = []  # (original, presidio_token, numbered)
    type_counter: dict[str, int] = defaultdict(int)
    for entity in entity_order:
        type_counter[entity.entity_type] += 1
        n = type_counter[entity.entity_type]
        placeholder = f"[{entity.entity_type}_{n}]"
        occurrence_mapping.append((entity.original, entity.placeholder, placeholder))
        mapping[placeholder] = entity.original

    # Replace occurrences of <TYPE> in the redacted text left-to-right.
    # Each <TYPE> tag corresponds to one entity in document order.
    result = redacted.redacted
    type_seen: dict[str, int] = defaultdict(int)
    for original, presidio_token, numbered_placeholder in occurrence_mapping:
        result = result.replace(presidio_token, numbered_placeholder, 1)

    return result, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text
