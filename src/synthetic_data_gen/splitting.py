"""Exact grouped train/eval splitting."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from synthetic_data_gen.labels import LABELS
from synthetic_data_gen.schema import RouterExample


def split_exact_by_route(
    rows: Sequence[RouterExample],
    *,
    train_size: int,
    eval_size: int,
    seed: int,
) -> tuple[list[RouterExample], list[RouterExample]]:
    if train_size % len(LABELS) != 0 or eval_size % len(LABELS) != 0:
        raise ValueError("train_size and eval_size must be divisible by the number of labels.")

    train_quota = train_size // len(LABELS)
    eval_quota = eval_size // len(LABELS)
    rng = random.Random(seed)

    by_route: dict[str, list[RouterExample]] = defaultdict(list)
    for row in rows:
        by_route[row.route].append(row)
    for route_rows in by_route.values():
        route_rows.sort(key=lambda row: row.id or "")
        rng.shuffle(route_rows)

    eval_groups: set[str] = set()
    train_groups: set[str] = set()
    eval_rows: list[RouterExample] = []
    train_rows: list[RouterExample] = []

    for route in LABELS:
        route_eval: list[RouterExample] = []
        for row in by_route.get(route, []):
            if row.group_key in train_groups:
                continue
            eval_groups.add(row.group_key)
            route_eval.append(row)
            if len(route_eval) == eval_quota:
                break
        if len(route_eval) < eval_quota:
            raise ValueError(
                f"Not enough eval examples for {route}: need {eval_quota}, got {len(route_eval)}"
            )
        eval_rows.extend(route_eval)

    for route in LABELS:
        route_train: list[RouterExample] = []
        for row in by_route.get(route, []):
            if row.group_key in eval_groups:
                continue
            train_groups.add(row.group_key)
            route_train.append(row)
            if len(route_train) == train_quota:
                break
        if len(route_train) < train_quota:
            raise ValueError(
                f"Not enough train examples for {route}: need {train_quota}, got {len(route_train)}"
            )
        train_rows.extend(route_train)

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def validate_exact_splits(
    *,
    train: Sequence[RouterExample],
    eval_rows: Sequence[RouterExample],
    train_size: int,
    eval_size: int,
) -> dict[str, int]:
    expected_train = train_size // len(LABELS)
    expected_eval = eval_size // len(LABELS)
    if len(train) != train_size:
        raise ValueError(f"Expected {train_size} train rows, got {len(train)}")
    if len(eval_rows) != eval_size:
        raise ValueError(f"Expected {eval_size} eval rows, got {len(eval_rows)}")

    train_counts = Counter(row.route for row in train)
    eval_counts = Counter(row.route for row in eval_rows)
    for route in LABELS:
        if train_counts[route] != expected_train:
            raise ValueError(
                f"Train route {route} expected {expected_train}, got {train_counts[route]}"
            )
        if eval_counts[route] != expected_eval:
            raise ValueError(
                f"Eval route {route} expected {expected_eval}, got {eval_counts[route]}"
            )

    train_groups = {row.group_key for row in train}
    eval_groups = {row.group_key for row in eval_rows}
    leaked = train_groups & eval_groups
    if leaked:
        sample = ", ".join(sorted(leaked)[:5])
        raise ValueError(f"Group leakage detected between train and eval: {sample}")
    return {"group_leakage_count": len(leaked)}
