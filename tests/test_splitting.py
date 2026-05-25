from __future__ import annotations

from collections import Counter

from synthetic_data_gen.labels import LABELS
from synthetic_data_gen.splitting import split_exact_by_route, validate_exact_splits
from synthetic_data_gen.types import (
    CompanyName,
    GeneratedText,
    GroupKey,
    RouteName,
    RouterExample,
    SourceName,
)


def make_example(route: RouteName, group: GroupKey, index: int) -> RouterExample:
    return RouterExample(
        text=GeneratedText(f"Finance prompt {route} {group} {index}"),
        route=route,
        source=SourceName("unit"),
        company=CompanyName("Example Inc"),
        metadata={"group_key": group},
    )


def test_split_exact_by_route_has_counts_and_no_leakage() -> None:
    rows = [
        make_example(route, GroupKey(f"group:{route}:{group_index}"), row_index)
        for route in LABELS
        for group_index in range(6)
        for row_index in range(2)
    ]

    train, eval_rows = split_exact_by_route(rows, train_size=10, eval_size=5, seed=7)
    leakage = validate_exact_splits(
        train=train,
        eval_rows=eval_rows,
        train_size=10,
        eval_size=5,
    )

    assert Counter(row.route for row in train) == {route: 2 for route in LABELS}
    assert Counter(row.route for row in eval_rows) == {route: 1 for route in LABELS}
    assert leakage == {"group_leakage_count": 0}
