"""Institution personas used to vary synthetic prompt style."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    institution_type: str
    style_hint: str


PERSONAS: tuple[Persona, ...] = (
    Persona("equity research analyst", "sell-side research", "concise, company-specific"),
    Persona("investment banking associate", "investment bank", "deal-oriented and precise"),
    Persona("hedge fund analyst", "hedge fund", "skeptical, catalyst-focused"),
    Persona("private equity associate", "private equity firm", "diligence-focused"),
    Persona("credit analyst", "credit fund", "risk and leverage focused"),
    Persona("corporate FP&A manager", "corporate finance team", "operational and budget-aware"),
    Persona("CFO office analyst", "CFO office", "executive-ready and metric-driven"),
    Persona("auditor", "audit firm", "evidence-oriented and control-aware"),
    Persona("risk officer", "financial risk function", "risk and exposure focused"),
    Persona("rating agency analyst", "rating agency", "debt capacity and outlook focused"),
    Persona("regulator/examiner", "financial regulator", "compliance and disclosure focused"),
    Persona("portfolio manager", "asset manager", "allocation and performance focused"),
    Persona("quant researcher", "quant fund", "factor and time-series aware"),
    Persona(
        "investor relations analyst",
        "investor relations team",
        "shareholder-communication aware",
    ),
)


def select_persona(rng: random.Random) -> Persona:
    return PERSONAS[rng.randrange(len(PERSONAS))]
