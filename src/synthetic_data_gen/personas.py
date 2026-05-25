"""Institution personas used to vary synthetic prompt style."""

from __future__ import annotations

import random
from dataclasses import dataclass

from synthetic_data_gen.types import InstitutionType, PersonaName, StyleHint


@dataclass(frozen=True)
class Persona:
    name: PersonaName
    institution_type: InstitutionType
    style_hint: StyleHint


PERSONAS: tuple[Persona, ...] = (
    Persona(
        PersonaName("equity research analyst"),
        InstitutionType("sell-side research"),
        StyleHint("concise, company-specific"),
    ),
    Persona(
        PersonaName("investment banking associate"),
        InstitutionType("investment bank"),
        StyleHint("deal-oriented and precise"),
    ),
    Persona(
        PersonaName("hedge fund analyst"),
        InstitutionType("hedge fund"),
        StyleHint("skeptical, catalyst-focused"),
    ),
    Persona(
        PersonaName("private equity associate"),
        InstitutionType("private equity firm"),
        StyleHint("diligence-focused"),
    ),
    Persona(
        PersonaName("credit analyst"),
        InstitutionType("credit fund"),
        StyleHint("risk and leverage focused"),
    ),
    Persona(
        PersonaName("corporate FP&A manager"),
        InstitutionType("corporate finance team"),
        StyleHint("operational and budget-aware"),
    ),
    Persona(
        PersonaName("CFO office analyst"),
        InstitutionType("CFO office"),
        StyleHint("executive-ready and metric-driven"),
    ),
    Persona(
        PersonaName("auditor"),
        InstitutionType("audit firm"),
        StyleHint("evidence-oriented and control-aware"),
    ),
    Persona(
        PersonaName("risk officer"),
        InstitutionType("financial risk function"),
        StyleHint("risk and exposure focused"),
    ),
    Persona(
        PersonaName("rating agency analyst"),
        InstitutionType("rating agency"),
        StyleHint("debt capacity and outlook focused"),
    ),
    Persona(
        PersonaName("regulator/examiner"),
        InstitutionType("financial regulator"),
        StyleHint("compliance and disclosure focused"),
    ),
    Persona(
        PersonaName("portfolio manager"),
        InstitutionType("asset manager"),
        StyleHint("allocation and performance focused"),
    ),
    Persona(
        PersonaName("quant researcher"),
        InstitutionType("quant fund"),
        StyleHint("factor and time-series aware"),
    ),
    Persona(
        PersonaName("investor relations analyst"),
        InstitutionType("investor relations team"),
        StyleHint("shareholder-communication aware"),
    ),
)


def select_persona(rng: random.Random) -> Persona:
    return PERSONAS[rng.randrange(len(PERSONAS))]
