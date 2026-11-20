"""Request-local source provenance, independent of benchmark gold annotations.

Spans reference the rendered tool result, so preview clipping cannot accidentally
attribute headers, arguments, or the omitted middle of a result to source text.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def source_observation(
    uri: str, *, content_kind: str = "body", success: bool = True, **kwargs: Any
) -> dict[str, Any]:
    return {
        "uri": uri,
        "content_kind": content_kind,
        "success": success,
        "returned_fragments": [],
        "delivered_fragments": None,
        **kwargs,
        "_spans": [],
    }


def append_source_fragment(
    lines: list[str], source: dict[str, Any], text: str, *, prefix: str = ""
) -> None:
    """Append a contiguous source fragment while retaining its rendered offset."""
    start = sum(len(line) + 1 for line in lines) + len(prefix)
    lines.append(prefix + text)
    if text:
        source["returned_fragments"].append({"text": text})
        source["_spans"].append((start, start + len(text), False))


def capture_json_sources(
    text: str, items: list[dict[str, Any]], *, field: str, kind: str
) -> list[dict[str, Any]]:
    """Capture source string values from the JSON we just rendered from items."""
    sources = []
    cursor = 0
    for item in items:
        if not item.get("uri"):
            continue
        success = item.get("success", True)
        source = source_observation(
            item["uri"], content_kind=kind, success=success,
            content_hash=item.get("base_hash"),
            is_directory=not item.get("is_leaf", True),
        )
        value = item.get(field, "")
        if isinstance(value, str):
            encoded = json.dumps(value, ensure_ascii=False)
            # Match the field and its value, not an occurrence in a URI/header.
            marker = f'{json.dumps(field)}: {encoded}'
            index = text.find(marker, cursor)
            if index < 0:
                raise ValueError("Rendered JSON source field could not be located")
            start = index + len(marker) - len(encoded) + 1
            cursor = start + len(encoded)
            if success and value:
                source["returned_fragments"].append({"text": value})
                source["_spans"].append((start, start + len(encoded) - 2, True))
        sources.append(source)
    return sources


def _decode_json_piece(encoded: str, start: int, end: int) -> str:
    """Keep only complete JSON character encodings within the visible range."""
    pieces = []
    index = 0
    while index < len(encoded):
        next_index = index + 1
        if encoded[index] == "\\":
            next_index = index + (6 if encoded[index + 1] == "u" else 2)
        if index >= start and next_index <= end:
            pieces.append(json.loads('"' + encoded[index:next_index] + '"'))
        index = next_index
    return "".join(pieces)


def delivered_sources(
    sources: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    original: str,
    delivered: str,
) -> list[dict[str, Any]]:
    """Project explicit source spans onto the standard tool preview.

    Unknown transformations leave delivery uncaptured. Post-call hooks that
    replace output must clear captures before reaching this function.
    """
    if delivered == original or delivered.startswith(original + "\n\nMedia content was omitted"):
        ranges = [(0, len(original))]
    elif (
        len(original) > 8_000
        and delivered.startswith(original[:4_000] + "\n\n[Output truncated. ")
        and delivered.endswith("]\n\n" + original[-2_000:])
    ):
        ranges = [(0, 4_000), (len(original) - 2_000, len(original))]
    else:
        ranges = None
    result = []
    for captured in sources:
        source = copy.deepcopy(captured)
        spans = source.pop("_spans", [])
        if ranges is not None:
            source["delivered_fragments"] = []
            for start, end, is_json in spans:
                for left, right in ranges:
                    lo, hi = max(start, left), min(end, right)
                    if lo >= hi:
                        continue
                    value = (
                        _decode_json_piece(original[start:end], lo - start, hi - start)
                        if is_json else original[lo:hi]
                    )
                    if value:
                        source["delivered_fragments"].append({"text": value})
            source["truncated"] = source.get("truncated", False) or ranges != [
                (0, len(original))
            ]
        result.append(source)
    return result
