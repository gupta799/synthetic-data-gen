"""Prompts for schema-backed finance router data generation."""

from __future__ import annotations

from synthetic_data_gen.labels import LABELS
from synthetic_data_gen.personas import Persona
from synthetic_data_gen.types import PromptText, RouteName, SeedRecord

SYSTEM_PROMPT = PromptText("""You generate training data for a finance-only LLM router.

The classifier will see a finance prompt and choose exactly one route:
- metric_extraction
- filing_summarization
- financial_qa
- financial_reasoning
- comparative_analysis

Generate realistic user requests from financial institution personas. The output must be strict
JSON only. Do not include markdown, commentary, answers, labels in the text, classifier language,
synthetic-data language, or phrases like "as an AI".

The generated `text` must be the prompt a real finance user would send to an assistant. It should
ask for work, not provide the work. It should not reveal the route name.
""")

ROUTE_INSTRUCTIONS: dict[RouteName, PromptText] = {
    RouteName("metric_extraction"): PromptText(
        "The prompt asks for a specific financial number, KPI, line item, ratio, table value, "
        "or period-specific metric. It should sound like a request to extract or identify a value."
    ),
    RouteName("filing_summarization"): PromptText(
        "The prompt asks to summarize, brief, condense, or explain the key points from a filing, "
        "10-K, 10-Q, annual report, MD&A, note, risk section, or excerpt."
    ),
    RouteName("financial_qa"): PromptText(
        "The prompt asks a straightforward factual finance or filing question that does not "
        "require multi-step reasoning, comparison, or direct metric extraction."
    ),
    RouteName("financial_reasoning"): PromptText(
        "The prompt asks for drivers, why/how analysis, financial implications, calculation, ratio "
        "interpretation, valuation context, capital intensity, trend explanation, or judgment."
    ),
    RouteName("comparative_analysis"): PromptText(
        "The prompt compares companies, periods, segments, sectors, metrics, filings, or excerpts. "
        "It should ask for relative performance, stronger profile, better margin, trend "
        "comparison, or side-by-side analysis."
    ),
}


def build_generation_prompt(
    *,
    route: RouteName,
    persona: Persona,
    seed: SeedRecord,
    batch_size: int,
) -> PromptText:
    route_list = ", ".join(LABELS)
    return PromptText(f"""Generate {batch_size} diverse finance user prompts.

Target route: {route}
Allowed routes: {route_list}
Route instruction: {ROUTE_INSTRUCTIONS[route]}

Persona:
- role: {persona.name}
- institution type: {persona.institution_type}
- style hint: {persona.style_hint}

Grounding seed:
- source: {seed.source}
- group: {seed.group_key}
- company: {seed.company or "unknown"}
- document type: {seed.document_type or "unknown"}
- period: {seed.period or "unknown"}
- context: {seed.short_context}

Rules:
- Return a JSON object with an `items` array containing exactly {batch_size} objects.
- Each object must have: text, route, company, persona, institution_type, metadata.
- route must be exactly "{route}".
- persona must be exactly "{persona.name}".
- institution_type must be exactly "{persona.institution_type}".
- company should use the seed company when available.
- metadata must be an object with seed_group, seed_source, route_instruction, and style_hint.
- The text should be one natural user prompt, 12 to 90 words.
- Do not write answers, bullets, explanations, labels, or markdown.
- Do not include the route name in the generated text.
- Make the prompts diverse in wording, financial intent, and user phrasing.
""")
