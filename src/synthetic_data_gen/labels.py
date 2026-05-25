"""Finance router labels shared by generation and validation."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "metric_extraction",
    "filing_summarization",
    "financial_qa",
    "financial_reasoning",
    "comparative_analysis",
)

LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}


def validate_route(route: str) -> str:
    if route not in LABEL_TO_ID:
        raise ValueError(f"Unknown route {route!r}; expected one of {', '.join(LABELS)}")
    return route
