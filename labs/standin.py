"""A scripted stand-in for the two hosted models, for a room without a working key.

It is not a model. It looks the request up in data/standin/index.json, takes the right tool
call, and then makes the mistakes a prompted model makes, at rates that depend on what the
prompt contains: a schema, the activity code table, rules for escalation, examples. A given
request fails in the same way on every run, and a better prompt removes mistakes without
adding new ones, so a lab's numbers are stable.

What its numbers mean: nothing about any real model. They let every cell run and every method
be practised. Run the labs against the hosted models to get numbers worth reporting.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import chamber as C

_INDEX: dict | None = None
_TO_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# probability of each kind of mistake: (with no help from the prompt, with the help that addresses it, with examples too)
RATES = {
    "small": {"escalate": (0.75, 0.55, 0.45), "scope": (0.55, 0.35, 0.30), "distractor": (0.50, 0.50, 0.42), "tool": (0.14, 0.14, 0.09),
              "years": (0.10, 0.10, 0.06), "knowledge": (0.85, 0.07, 0.07), "indic": (0.65, 0.45, 0.40), "format": (0.26, 0.09, 0.06),
              "name": (0.18, 0.18, 0.12)},
    "large": {"escalate": (0.30, 0.05, 0.04), "scope": (0.12, 0.03, 0.02), "distractor": (0.10, 0.10, 0.04), "tool": (0.02, 0.02, 0.01),
              "years": (0.01, 0.01, 0.01), "knowledge": (0.80, 0.02, 0.02), "indic": (0.15, 0.03, 0.02), "format": (0.04, 0.01, 0.01),
              "name": (0.03, 0.03, 0.02)},
}


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def index() -> dict:
    global _INDEX
    if _INDEX is None:
        path = Path(__file__).resolve().parent / "data" / "standin" / "index.json"
        _INDEX = json.loads(path.read_text(encoding="utf-8"))
    return _INDEX


def prompt_features(messages: list[dict]) -> dict:
    """What the instructions and earlier turns give the model, as the stand-in reads them."""
    before = "\n".join(str(m.get("content") or "") for m in messages[:-1])
    low = before.lower()
    return {
        "schema": "cr_number" in before and "arguments" in before,
        "digits": "western digits" in low,
        "table": ("retail trade" in low and "agriculture" in low) or "تجارة التجزئة" in before,
        "rules": "escalate_to_staff" in before and "missing" in low,
        "examples": before.count("<tool_call>") >= 2,
    }


def _uniform(rid: str, role: str, kind: str) -> float:
    h = hashlib.sha256(f"{rid}|{role}|{kind}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _rate(role: str, kind: str, helped: bool, examples: bool) -> float:
    none, with_help, with_examples = RATES[role][kind]
    if helped and examples:
        return with_examples
    if helped:
        return with_help
    if examples:
        return max(with_examples, (none + with_examples) / 2)  # examples alone help half as much
    return none


def simulate(entry: dict, role: str, feats: dict) -> tuple[str, list[str]]:
    """The stand-in's answer to one request, and the mistakes it made (for the answer key)."""
    rid, lang, tags = entry["id"], entry["lang"], entry["tags"]
    gold = entry["gold"]
    call = {"name": gold["name"], "arguments": dict(gold["arguments"])}
    rng = random.Random(f"{rid}|{role}|variant")
    ex = feats["examples"]
    made = []

    def happens(kind: str, helped: bool, factor: float = 1.0) -> bool:
        return _uniform(rid, role, kind) < _rate(role, kind, helped, ex) * factor

    # behaviour
    if "missing_cr" in tags and happens("escalate", feats["rules"]):
        call = C.call("renew_membership", cr_number="4000000000", years=1)
        made.append("no_escalation")
    elif "out_of_scope" in tags and happens("scope", feats["rules"]):
        made.append("answered_out_of_scope")
        text = "I am happy to help with that. Please send the details and the chamber will arrange it." if lang == "en" else "يسعدني مساعدتكم في ذلك. نرجو إرسال التفاصيل وستتولى الغرفة الأمر."
        return text, made
    else:
        if "distractor_cr" in tags and happens("distractor", True):
            call["arguments"]["cr_number"] = entry["distractor"]
            made.append("wrong_registration_number")
        if call["name"] in ("renew_membership", "issue_certificate", "update_record") and happens("tool", True):
            cr = call["arguments"]["cr_number"]
            call = C.call("issue_certificate", cr_number=cr, kind="membership") if call["name"] != "issue_certificate" else C.call("renew_membership", cr_number=cr, years=1)
            made.append("wrong_tool")
        if call["name"] == "renew_membership" and "wrong_tool" not in made and happens("years", True):
            call["arguments"]["years"] = {1: 2, 2: 1, 3: 2}[call["arguments"]["years"]]
            made.append("wrong_years")
        if lang == "ar" and call["name"] == "create_membership" and happens("name", True):
            call["arguments"]["company_name"] = " ".join(call["arguments"]["company_name"].split()[1:])
            made.append("name_not_copied")
    # knowledge: the chamber's own activity codes
    if "activity" in tags and "wrong_tool" not in made and happens("knowledge", feats["table"]):
        key = "activity_code" if call["name"] == "create_membership" else "value"
        if rng.random() < 0.5:
            call["arguments"][key] = rng.choice(["ACT-47", "4711", "G-46", "IT-62", "SIC-5411"])
        else:
            call["arguments"][key] = rng.choice([c for c in C.ACTIVITY_CODES if c != call["arguments"][key]])
        made.append("wrong_activity_code")
    # format
    if "indic_digits" in tags and "cr_number" in call["arguments"] and happens("indic", feats["schema"] and feats["digits"]):
        call["arguments"]["cr_number"] = call["arguments"]["cr_number"].translate(_TO_INDIC)
        made.append("digits_not_normalised")
    body = json.dumps(call, ensure_ascii=False)
    if happens("format", feats["schema"], 1.4 if (lang == "ar" and role == "small") else 1.0):
        fault = rng.choice(["prose", "fence", "python_dict", "keys", "cut"])
        made.append("format_" + fault)
        if fault == "prose":
            args = ", ".join(f"{k} {v}" for k, v in call["arguments"].items())
            return f"This request needs {call['name']} with {args}.", made
        if fault == "fence":
            return "```json\n" + body + "\n```", made
        if fault == "python_dict":
            return "<tool_call>\n" + str(call) + "\n</tool_call>", made
        if fault == "keys":
            renamed = {("cr" if k == "cr_number" else "name" if k == "company_name" else k): v for k, v in call["arguments"].items()}
            return "<tool_call>\n" + json.dumps({"tool": call["name"], "args": renamed}, ensure_ascii=False) + "\n</tool_call>", made
        return "<tool_call>\n" + body[: max(10, len(body) - 9)], made
    return "<tool_call>\n" + body + "\n</tool_call>", made


def answer(messages: list[dict], role: str = "small") -> str:
    """The stand-in's reply to a conversation whose last message is the request."""
    request = str(messages[-1].get("content") or "")
    entry = index().get(text_key(request))
    if entry is None:
        return "The scripted stand-in only knows the course's requests. Set a key to ask a hosted model."
    return simulate(entry, role, prompt_features(messages))[0]
