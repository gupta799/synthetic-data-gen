"""Colored terminal output for long-running builds."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from rich.console import Console
from rich.theme import Theme

from synthetic_data_gen.types import GeneratedText

CONSOLE = Console(
    theme=Theme(
        {
            "info": "cyan",
            "success": "bold green",
            "warning": "yellow",
            "label": "bold white",
            "value": "white",
        }
    )
)


def _print_fields(fields: Mapping[GeneratedText, object] | None) -> None:
    for key, value in (fields or {}).items():
        display = str(value) if not isinstance(value, Path) else value.as_posix()
        CONSOLE.print(f"  [label]{key}[/label]: [value]{display}[/value]")


def log_info(message: GeneratedText, fields: Mapping[GeneratedText, object] | None = None) -> None:
    CONSOLE.print(f"[info]{message}[/info]")
    _print_fields(fields)


def log_success(
    message: GeneratedText,
    fields: Mapping[GeneratedText, object] | None = None,
) -> None:
    CONSOLE.print(f"[success]{message}[/success]")
    _print_fields(fields)


def log_warning(
    message: GeneratedText,
    fields: Mapping[GeneratedText, object] | None = None,
) -> None:
    CONSOLE.print(f"[warning]{message}[/warning]")
    _print_fields(fields)
