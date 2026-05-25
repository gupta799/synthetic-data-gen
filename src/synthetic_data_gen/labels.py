"""Finance router labels shared by generation and validation."""

from __future__ import annotations

from synthetic_data_gen.types.domain import RouteName

LABELS: tuple[RouteName, ...] = (
    RouteName("metric_extraction"),
    RouteName("filing_summarization"),
    RouteName("financial_qa"),
    RouteName("financial_reasoning"),
    RouteName("comparative_analysis"),
)

LABEL_TO_ID: dict[RouteName, int] = {label: index for index, label in enumerate(LABELS)}


def validate_route(route: object) -> RouteName:
    route_name = RouteName(str(route))
    if route_name not in LABEL_TO_ID:
        raise ValueError(f"Unknown route {route_name!r}; expected one of {', '.join(LABELS)}")
    return route_name
