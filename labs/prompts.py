"""The instructions Qayd was piloted with, in parts, so a lab can add one part at a time."""

from __future__ import annotations

import chamber as C

# Version 0: what the pilot started with. No schema, no code table, no rules, no examples.
V0 = "You are Qayd, the registration assistant of Al-Bayraq Chamber of Commerce. Turn each request into a call to the registry."


def signatures() -> str:
    """The tool interface in one line per tool: names and argument keys, without their descriptions."""
    lines = []
    for tool in C.TOOLS:
        f = tool["function"]
        lines.append(f"{f['name']}({', '.join(f['parameters']['properties'])})")
    return "\n".join(lines)


SCHEMA_HINT = (
    "Answer with exactly one tool call and nothing else, in this form:\n"
    '<tool_call>\n{"name": "<tool>", "arguments": {...}}\n</tool_call>\n'
    "Tools:\n" + signatures() + "\n"
    "cr_number is 10 Western digits (convert Arabic-Indic digits). company_name is copied as written. city is the English name. years is an integer."
)

CODE_TABLE = "Activity codes of the chamber (revision 2026):\n" + "\n".join(f"{a[0]} {a[1]} / {a[2]}" for a in C.ACTIVITIES)

RULES = (
    "Rules:\n"
    "- If the request needs a registration number and none is given, call escalate_to_staff with reason missing_cr. Never invent a number.\n"
    "- If the request is not about membership, a record or a certificate, call escalate_to_staff with reason out_of_scope.\n"
    "- When two registration numbers appear, use the applicant's own."
)


def build(schema: bool = False, table: bool = False, rules: bool = False) -> str:
    """Instructions made of version 0 and the parts asked for."""
    parts = [V0]
    if schema:
        parts.append(SCHEMA_HINT)
    if table:
        parts.append(CODE_TABLE)
    if rules:
        parts.append(RULES)
    return "\n\n".join(parts)


# The instructions frozen as the baseline at the end of Day 1, used for the pilot traces and the teacher.
FROZEN = build(schema=True, table=True, rules=True)

# Four worked examples shown before the request (from the validation requests, never from the test set).
EXAMPLE_IDS = ["V-0013", "V-0006", "V-0014", "V-0011"]
