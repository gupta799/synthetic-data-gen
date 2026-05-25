from __future__ import annotations

from synthetic_data_gen.parser import parse_json_objects
from synthetic_data_gen.types import RawModelOutput


def test_parse_json_objects_from_fenced_array() -> None:
    rows = parse_json_objects(
        RawModelOutput("""```json
        [{"text": "What was revenue?", "route": "metric_extraction"}]
        ```""")
    )

    assert rows == [{"text": "What was revenue?", "route": "metric_extraction"}]


def test_parse_json_objects_wraps_single_object() -> None:
    rows = parse_json_objects(
        RawModelOutput('{"text": "Summarize the filing.", "route": "filing_summarization"}')
    )

    assert len(rows) == 1
    assert rows[0]["route"] == "filing_summarization"
