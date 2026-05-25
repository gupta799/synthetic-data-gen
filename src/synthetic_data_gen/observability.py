"""Local, LangSmith, and W&B observability for dataset builds."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter

from synthetic_data_gen.types import (
    ArtifactName,
    ArtifactType,
    EventName,
    JsonObject,
    MetricPayload,
    ProjectName,
    RunName,
)


def jsonable(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    return value


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started = perf_counter()
        self._handle = self.path.open("w", encoding="utf-8")

    def log(self, event: EventName, **payload: object) -> None:
        row = {
            "event": event,
            "elapsed_seconds": perf_counter() - self._started,
            **{key: jsonable(value) for key, value in payload.items()},
        }
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class WandbLogger:
    def __init__(
        self,
        *,
        project: ProjectName | None,
        run_name: RunName | None,
        config: JsonObject,
    ) -> None:
        self.run = None
        if not project:
            return
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            mode=os.getenv("WANDB_MODE", "online"),
            config=config,
        )

    def log(self, payload: MetricPayload, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(payload, step=step)

    def log_artifact(self, name: ArtifactName, artifact_type: ArtifactType, path: Path) -> None:
        if self.run is None:
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        artifact.add_dir(str(path))
        self.run.log_artifact(artifact)

    def finish(self, summary: JsonObject | None = None) -> None:
        if self.run is None:
            return
        for key, value in (summary or {}).items():
            self.run.summary[key] = value
        self.run.finish()


def configure_langsmith(project: ProjectName | None) -> None:
    if not project:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
