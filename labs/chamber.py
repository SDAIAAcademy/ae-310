"""Al-Bayraq Chamber of Commerce: the vocabulary of the case study and the registry tool interface.

Everything here is fictional and synthetic. The module has no dependency outside the standard
library, so the data generator, the labs and the server can all import it.
"""

from __future__ import annotations

import json
import re

CHAMBER = "Al-Bayraq Chamber of Commerce"
ASSISTANT = "Qayd"

CITIES = [
    ("Riyadh", "الرياض"), ("Jeddah", "جدة"), ("Dammam", "الدمام"), ("Khobar", "الخبر"),
    ("Abha", "أبها"), ("Tabuk", "تبوك"), ("Hail", "حائل"), ("Buraidah", "بريدة"),
    ("Taif", "الطائف"), ("Jazan", "جازان"), ("Najran", "نجران"), ("Yanbu", "ينبع"),
]
CITY_EN = [c for c, _ in CITIES]
CITY_AR = dict(CITIES)

# The chamber's own activity codes (revision 2026). A model cannot know them: they are knowledge.
ACTIVITIES = [
    ("BR-101", "retail trade", "تجارة التجزئة", ["retail trade", "retail shops"], ["تجارة التجزئة", "البيع بالتجزئة"]),
    ("BR-102", "wholesale trade", "تجارة الجملة", ["wholesale trade", "wholesale distribution"], ["تجارة الجملة", "التوزيع بالجملة"]),
    ("BR-210", "building materials", "مواد البناء", ["building materials", "construction supplies"], ["مواد البناء", "لوازم البناء"]),
    ("BR-215", "contracting", "المقاولات", ["contracting", "general contracting"], ["المقاولات", "المقاولات العامة"]),
    ("BR-320", "restaurants and catering", "المطاعم والتموين", ["restaurants and catering", "catering services"], ["المطاعم والتموين", "خدمات الإعاشة"]),
    ("BR-330", "food manufacturing", "الصناعات الغذائية", ["food manufacturing", "food production"], ["الصناعات الغذائية", "إنتاج الأغذية"]),
    ("BR-410", "transport and logistics", "النقل والخدمات اللوجستية", ["transport and logistics", "freight services"], ["النقل والخدمات اللوجستية", "خدمات الشحن"]),
    ("BR-520", "information technology", "تقنية المعلومات", ["information technology", "software services"], ["تقنية المعلومات", "خدمات البرمجيات"]),
    ("BR-530", "consulting", "الاستشارات", ["consulting", "business consulting"], ["الاستشارات", "الاستشارات الإدارية"]),
    ("BR-610", "medical supplies", "المستلزمات الطبية", ["medical supplies", "medical equipment"], ["المستلزمات الطبية", "الأجهزة الطبية"]),
    ("BR-720", "tourism and travel", "السياحة والسفر", ["tourism and travel", "travel services"], ["السياحة والسفر", "خدمات السفر"]),
    ("BR-810", "agriculture", "الزراعة", ["agriculture", "farm produce"], ["الزراعة", "المنتجات الزراعية"]),
]
ACTIVITY_CODES = [a[0] for a in ACTIVITIES]

CERTIFICATE_KINDS = ["membership", "origin"]
UPDATE_FIELDS = ["city", "activity_code"]
ESCALATION_REASONS = ["missing_cr", "out_of_scope"]

# The registry tool interface, in the JSON Schema form that chat templates and hosted APIs take.
TOOLS = [
    {"type": "function", "function": {
        "name": "create_membership",
        "description": "Register a company as a new member of the chamber.",
        "parameters": {"type": "object", "properties": {
            "company_name": {"type": "string", "description": "The company name exactly as written in the request"},
            "cr_number": {"type": "string", "description": "Commercial registration number: 10 Western digits"},
            "city": {"type": "string", "enum": CITY_EN},
            "activity_code": {"type": "string", "enum": ACTIVITY_CODES}},
            "required": ["company_name", "cr_number", "city", "activity_code"]}}},
    {"type": "function", "function": {
        "name": "renew_membership",
        "description": "Renew an existing membership for a number of years.",
        "parameters": {"type": "object", "properties": {
            "cr_number": {"type": "string", "description": "Commercial registration number: 10 Western digits"},
            "years": {"type": "integer", "enum": [1, 2, 3]}},
            "required": ["cr_number", "years"]}}},
    {"type": "function", "function": {
        "name": "update_record",
        "description": "Change one field of a member's record.",
        "parameters": {"type": "object", "properties": {
            "cr_number": {"type": "string", "description": "Commercial registration number: 10 Western digits"},
            "field": {"type": "string", "enum": UPDATE_FIELDS},
            "value": {"type": "string", "description": "A city name in English, or an activity code"}},
            "required": ["cr_number", "field", "value"]}}},
    {"type": "function", "function": {
        "name": "issue_certificate",
        "description": "Issue a membership certificate or a certificate of origin.",
        "parameters": {"type": "object", "properties": {
            "cr_number": {"type": "string", "description": "Commercial registration number: 10 Western digits"},
            "kind": {"type": "string", "enum": CERTIFICATE_KINDS}},
            "required": ["cr_number", "kind"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_staff",
        "description": "Hand the request to the registry desk when a required field is missing or the request is outside the registry.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "enum": ESCALATION_REASONS}},
            "required": ["reason"]}}},
]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
ARGUMENT_ORDER = {t["function"]["name"]: list(t["function"]["parameters"]["properties"]) for t in TOOLS}

_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def western_digits(text: str) -> str:
    """Arabic-Indic digits to Western digits."""
    return text.translate(_INDIC)


def call(name: str, **arguments) -> dict:
    """A tool call with its arguments in the interface's order."""
    return {"name": name, "arguments": {k: arguments[k] for k in ARGUMENT_ORDER[name] if k in arguments}}


def render_call(tool_call: dict) -> str:
    """The text an assistant turn holds for a tool call: one JSON object in tool_call tags."""
    return "<tool_call>\n" + json.dumps(tool_call, ensure_ascii=False) + "\n</tool_call>"


_TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_call(text: str) -> dict | None:
    """The tool call inside a model's output, or None when there is none that parses.

    Strict on purpose: one JSON object with "name" and "arguments", inside tool_call tags. A model
    that answers in prose, in a code fence or with a Python dict has not made a tool call.
    """
    if not isinstance(text, str):
        return None
    m = _TAGGED.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not isinstance(obj.get("arguments"), dict):
        return None
    return {"name": obj["name"], "arguments": obj["arguments"]}


def schema_errors(tool_call: dict | None) -> list[str]:
    """Why a parsed call does not fit the tool interface; an empty list when it fits."""
    if tool_call is None:
        return ["no tool call that parses"]
    name, args = tool_call["name"], tool_call["arguments"]
    if name not in TOOL_NAMES:
        return [f"unknown tool {name!r}"]
    spec = TOOLS[TOOL_NAMES.index(name)]["function"]["parameters"]
    errors = []
    for key in spec["required"]:
        if key not in args:
            errors.append(f"missing argument {key!r}")
    for key, value in args.items():
        prop = spec["properties"].get(key)
        if prop is None:
            errors.append(f"unknown argument {key!r}")
            continue
        if prop["type"] == "string" and not isinstance(value, str):
            errors.append(f"{key!r} is not a string")
        if prop["type"] == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
            errors.append(f"{key!r} is not an integer")
        if "enum" in prop and value not in prop["enum"]:
            errors.append(f"{key!r} has a value outside the list: {value!r}")
    cr = args.get("cr_number")
    if isinstance(cr, str) and not re.fullmatch(r"[0-9]{10}", cr):
        errors.append("'cr_number' is not 10 Western digits")
    if name == "update_record" and not errors:
        allowed = CITY_EN if args["field"] == "city" else ACTIVITY_CODES
        if args["value"] not in allowed:
            errors.append(f"'value' does not fit the field {args['field']!r}: {args['value']!r}")
    return errors


def same_call(a: dict | None, b: dict | None) -> bool:
    """Exact match of two tool calls: the same tool and the same arguments."""
    return a is not None and b is not None and a["name"] == b["name"] and a["arguments"] == b["arguments"]


_PHONE = re.compile(r"(?<![0-9])05[0-9]{8}(?![0-9])")
_NATIONAL_ID = re.compile(r"(?<![0-9])1[0-9]{9}(?![0-9])")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def find_personal_data(text: str, names=()) -> dict:
    """Personal data in a text, by kind: mobile numbers, national ID numbers, e-mail addresses, known names."""
    return {
        "phone": _PHONE.findall(text),
        "national_id": _NATIONAL_ID.findall(text),
        "email": _EMAIL.findall(text),
        "name": [n for n in names if n and n in text],
    }


def mask_personal_data(text: str, names=()) -> str:
    """Replace personal data with placeholders. Registration numbers (10 digits from 40) are not personal data."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _NATIONAL_ID.sub("[ID]", text)
    for n in names:
        if n:
            text = text.replace(n, "[NAME]")
    return text
