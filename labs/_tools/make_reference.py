"""Write the supplied states of the labs that need no training: traces, corrections, recorded
model outputs, the tuning sets and the reference results of Days 1 and 2.

    uv run python _tools/make_reference.py

Run after prepare_data.py. The model outputs here come from the scripted stand-in (standin.py):
the course was built without a working key. With a key, set ADAPT_RECORD=live and run again to
record the hosted models instead (about 2,300 requests; a free key does not allow that in a day).

This file holds the reference version of every Day 1 and Day 2 computation. The lab solutions
compute the same numbers on their own and assert that they agree.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

import chamber as C  # noqa: E402
import prepare_data as P  # noqa: E402
import prompts  # noqa: E402
import qayd as Q  # noqa: E402
import standin  # noqa: E402

if os.environ.get("ADAPT_RECORD", "standin") != "live":
    os.environ["ADAPT_STANDIN"] = "1"
    Q._STATE["standin"] = True

REF = Q.REFERENCE
CR = re.compile(r"(?<![0-9])40[0-9]{8}(?![0-9])")


def examples() -> list[dict]:
    val = {r["id"]: r for r in Q.load("requests_val")}
    return [{"text": val[i]["text"], "call": val[i]["gold"]} for i in prompts.EXAMPLE_IDS]


def outputs(rows, system, role, shots=None) -> list[str]:
    return [r["text"] for r in Q.run(rows, system, role, shots)]


# ------------------------------------------------------------------ Day 1

def failure_class(row: dict, output: str) -> str | None:
    """None when the output is the right call; otherwise knowledge, format or behaviour."""
    call = C.parse_call(output)
    if C.same_call(call, row["gold"]):
        return None
    if call is None:
        return "format"
    gold = row["gold"]
    key = "activity_code" if gold["name"] == "create_membership" else "value" if gold["arguments"].get("field") == "activity_code" else None
    if key and call["name"] == gold["name"] and key in call["arguments"]:
        patched = {"name": call["name"], "arguments": {**call["arguments"], key: gold["arguments"][key]}}
        if C.same_call(patched, gold):
            return "knowledge"
    if C.schema_errors(call):
        return "format"
    return "behaviour"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def day1() -> None:
    dev = Q.load("requests_100")
    table = {}
    for role in ("small", "large"):
        outs = outputs(dev, prompts.V0, role)
        Q.write_jsonl(REF / "outputs" / f"{role}_v0_100.jsonl", [{"id": r["id"], "output": o} for r, o in zip(dev, outs)])
        classes = Counter(failure_class(r, o) for r, o in zip(dev, outs))
        table[role] = {"correct": classes[None], "knowledge": classes["knowledge"], "format": classes["format"], "behaviour": classes["behaviour"]}
    Q.write_json(REF / "d1_1_decision_table.json", {
        "counts": table,
        "approach": {"knowledge": ["retrieval", "Both models fail it: the code table must be given, in context or by retrieval."],
                     "format": ["prompting", "The larger model rarely fails it: schema hints and examples first, tuning for what remains."],
                     "behaviour": ["distillation", "The larger model gets it right: teach the small model, by tuning on corrections or on the larger model's calls."]}})

    shots = examples()
    versions = [("v0", prompts.V0, None), ("v1", prompts.build(schema=True), None), ("v2", prompts.build(schema=True, table=True), None),
                ("v3", prompts.FROZEN, None), ("v4", prompts.FROZEN, shots)]
    scores = {}
    for name, system, sh in versions:
        outs = outputs(dev, system, "small", sh)
        classes = Counter(failure_class(r, o) for r, o in zip(dev, outs))
        k = classes[None]
        lo, hi = wilson(k, len(dev))
        scores[name] = {"correct": k, "n": len(dev), "accuracy": k / len(dev), "interval": [round(lo, 3), round(hi, 3)],
                        "knowledge": classes["knowledge"], "format": classes["format"], "behaviour": classes["behaviour"]}
    large = outputs(dev, prompts.FROZEN, "large", shots)
    scores["larger_model_v4"] = {"correct": sum(failure_class(r, o) is None for r, o in zip(dev, large)), "n": len(dev)}
    Q.write_json(REF / "d1_2_prompt_scores.json", scores)
    print("Day 1:", json.dumps(table), {k: v["correct"] for k, v in scores.items()})


# ------------------------------------------------------------------ Day 2

def to_example(rid: str, source: str, lang: str, text: str, call: dict) -> dict:
    """A tuning example in chat format: the request, then the assistant's tool call."""
    return {"id": rid, "source": source, "lang": lang, "messages": [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]}]}


def rule_errors(text: str, call: dict) -> list[str]:
    """Checks a schema cannot make: does the call agree with what the request says?"""
    errors = []
    numbers = CR.findall(C.western_digits(text))
    args = call["arguments"]
    if "cr_number" in args and args["cr_number"] not in numbers:
        errors.append("the registration number is not in the request")
    if call["name"] == "escalate_to_staff" and args.get("reason") == "missing_cr" and numbers:
        errors.append("escalated as missing although the request gives a number")
    if call["name"] == "create_membership" and args.get("company_name", "") not in text:
        errors.append("the company name is not in the request")
    city = args.get("city") or (args.get("value") if args.get("field") == "city" else None)
    if city and city not in text and C.CITY_AR.get(city, "\0") not in text:
        errors.append("the city is not in the request")
    return errors


def verify(text: str, output: str) -> list[str]:
    call = C.parse_call(output)
    errors = C.schema_errors(call)
    return errors if errors else rule_errors(text, call)


def normalise(text: str) -> list[str]:
    text = C.western_digits(text).lower()
    return re.sub(r"[^\w\s]", " ", text).split()


def shingles(text: str) -> set:
    w = normalise(text)
    return {" ".join(w[i : i + 3]) for i in range(max(1, len(w) - 2))}


def containment(a: set, b: set) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def bucket(text: str) -> str:
    numbers = CR.findall(C.western_digits(text))
    return numbers[0] if numbers else "no number"


def deduplicate(rows: list[dict], threshold: float = 0.9) -> tuple[list[dict], int]:
    kept, by_bucket, removed = [], {}, 0
    for row in rows:
        text = row["messages"][0]["content"]
        s, b = shingles(text), bucket(text)
        if any(containment(s, other) >= threshold for other in by_bucket.get(b, [])):
            removed += 1
            continue
        by_bucket.setdefault(b, []).append(s)
        kept.append(row)
    return kept, removed


def day2(b: dict) -> None:
    rng = random.Random(P.SEED + 2)
    shots = examples()
    traces = b["traces"]
    outs = outputs(traces, prompts.FROZEN, "small", shots)
    wrong_valid = [i for i, (r, o) in enumerate(zip(traces, outs)) if not C.same_call(C.parse_call(o), r["gold"]) and not C.schema_errors(C.parse_call(o))]
    accepted_wrong = set(rng.sample(wrong_valid, 8))  # planted: staff accepted eight wrong calls
    trace_rows, corrections = [], []
    for i, (r, o) in enumerate(zip(traces, outs)):
        right = C.same_call(C.parse_call(o), r["gold"])
        if right or i in accepted_wrong:
            status = "accepted"
        elif rng.random() < 0.8:
            status = "corrected"
            corrections.append({"trace_id": r["id"], "call": r["gold"], "by": "registry desk"})
        else:
            status = "abandoned"
        trace_rows.append({"id": r["id"], "lang": r["lang"], "text": r["text"], "output": o, "status": status})
    # planted: twelve corrections that are themselves wrong, in ways the tool interface rejects
    for n, j in enumerate(rng.sample(range(len(corrections)), 12)):
        call = json.loads(json.dumps(corrections[j]["call"]))
        args = call["arguments"]
        if "cr_number" in args and n % 3 == 0:
            args["cr_number"] = args["cr_number"][:9]
        elif "city" in args or args.get("field") == "city":
            key = "city" if "city" in args else "value"
            args[key] = C.CITY_AR[args[key]]
        elif "cr_number" in args:
            args["cr_number"] = args["cr_number"].translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
        else:
            args["reason"] = "no number"
        corrections[j]["call"] = call
    Q.write_jsonl(Q.DATA / "traces_500.jsonl", trace_rows)
    Q.write_jsonl(Q.DATA / "corrections.jsonl", corrections)

    # Lab 2.1: traces and corrections to examples
    by_id = {c["trace_id"]: c for c in corrections}
    ex_traces, dropped = [], Counter()
    for t in trace_rows:
        if t["status"] == "accepted":
            call = C.parse_call(t["output"])
            if C.schema_errors(call):
                dropped["accepted but not a valid call"] += 1
                continue
            ex_traces.append(to_example(t["id"], "trace", t["lang"], t["text"], call))
        elif t["status"] == "corrected":
            call = by_id[t["id"]]["call"]
            if C.schema_errors(call):
                dropped["correction rejected by the interface"] += 1
                continue
            ex_traces.append(to_example(t["id"], "correction", t["lang"], t["text"], call))
        else:
            dropped["abandoned: no label"] += 1
    gold = {r["id"]: r["gold"] for r in traces}
    noisy = sum(1 for e in ex_traces if not C.same_call(e["messages"][1]["tool_calls"][0]["function"], gold[e["id"]]))
    status = Counter(t["status"] for t in trace_rows)
    Q.write_jsonl(REF / "examples_traces.jsonl", ex_traces)
    d21 = {"status": dict(status), "corrections": len(corrections), "examples": len(ex_traces), "by_source": dict(Counter(e["source"] for e in ex_traces)),
           "dropped": dict(dropped), "wrong_labels_left": noisy}
    Q.write_json(REF / "d2_1_examples.json", d21)

    # Lab 2.2: the teacher labels the pool, a verifier filters
    pool = b["pool"]
    teacher = outputs(pool, prompts.FROZEN, "large", shots)
    Q.write_jsonl(REF / "outputs" / "teacher_pool.jsonl", [{"id": r["id"], "output": o} for r, o in zip(pool, teacher)])

    def filtered(rows, outs):
        kept, reasons, escaped = [], Counter(), 0
        for r, o in zip(rows, outs):
            errors = verify(r["text"], o)
            if errors:
                reasons["schema" if C.schema_errors(C.parse_call(o)) else "rule"] += 1
                continue
            call = C.parse_call(o)
            escaped += not C.same_call(call, r["gold"])
            kept.append(to_example(r["id"], "teacher", r["lang"], r["text"], call))
        return kept, reasons, escaped

    kept300, reasons300, escaped300 = filtered(pool[:300], teacher[:300])
    wrong300 = sum(not C.same_call(C.parse_call(o), r["gold"]) for r, o in zip(pool[:300], teacher[:300]))
    Q.write_jsonl(REF / "examples_teacher.jsonl", kept300)
    d22 = {"generated": 300, "teacher_wrong": wrong300, "rejected": dict(reasons300), "kept": len(kept300), "yield": round(len(kept300) / 300, 3),
           "wrong_but_kept": escaped300, "right_but_rejected": 300 - len(kept300) - (wrong300 - escaped300)}
    # the registry desk audits 30 kept examples by hand (here: against the generator's answers)
    pool_gold = {r["id"]: r["gold"] for r in pool}
    audit_rng = random.Random(P.SEED + 22)
    audit = [{"id": e["id"], "desk_says": "right" if C.same_call(e["messages"][1]["tool_calls"][0]["function"], pool_gold[e["id"]]) else "wrong"}
             for e in audit_rng.sample(kept300, 30)]
    Q.write_jsonl(REF / "audit_30.jsonl", audit)
    d22["audit_wrong_of_30"] = sum(a["desk_says"] == "wrong" for a in audit)
    Q.write_json(REF / "d2_2_teacher.json", d22)

    # Lab 2.4: deduplicate, decontaminate, mask
    raw = ex_traces + kept300
    Q.write_jsonl(REF / "examples_raw.jsonl", raw)
    names = sorted({r[k] for r in b["registry"] for k in ("owner_en", "owner_ar")})
    deduped, n_dups = deduplicate(raw)
    test = b["test"]
    test_by_bucket = {}
    for t in test:
        test_by_bucket.setdefault(bucket(t["text"]), []).append(shingles(t["text"]))
    clean, leaks, collisions = [], [], 0
    for row in deduped:
        text = row["messages"][0]["content"]
        s, bk = shingles(text), bucket(text)
        hit = any(containment(s, other) >= 0.9 for other in test_by_bucket.get(bk, []))
        if hit and bk != "no number":
            leaks.append(row["id"])
            continue
        collisions += hit
        clean.append(row)
    found = Counter()
    for row in clean:
        hits = C.find_personal_data(row["messages"][0]["content"], names)
        for kind, values in hits.items():
            found[kind] += len(values)
        row["messages"][0]["content"] = C.mask_personal_data(row["messages"][0]["content"], names)
    left = sum(len(v) for row in clean for v in C.find_personal_data(row["messages"][0]["content"], names).values())
    Q.write_jsonl(Q.DATA / "tuning" / "train.jsonl", clean)
    Q.write_jsonl(Q.DATA / "tuning" / "val.jsonl", [to_example(r["id"], "validation", r["lang"], C.mask_personal_data(r["text"], names), r["gold"]) for r in b["val"]])
    planted_leaks = {t["id"] for t in traces if t.get("leak_of")}
    d24 = {"rows_in": len(raw), "near_duplicates_removed": n_dups, "leaks_removed": len(leaks), "leaks_planted_in_traces": len(planted_leaks),
           "template_collisions_kept": collisions, "personal_data_found": dict(found), "personal_data_left": left, "rows_out": len(clean),
           "by_source": dict(Counter(r["source"] for r in clean)), "by_lang": dict(Counter(r["lang"] for r in clean)),
           "escalations": sum(r["messages"][1]["tool_calls"][0]["function"]["name"] == "escalate_to_staff" for r in clean)}
    Q.write_json(REF / "d2_4_clean.json", d24)

    # supplied sets for Day 3: all filtered teacher calls, and the corrections alone
    kept_all, _, escaped_all = filtered(pool, teacher)
    kept_all, _ = deduplicate(kept_all)
    for row in kept_all:
        row["messages"][0]["content"] = C.mask_personal_data(row["messages"][0]["content"], names)
    Q.write_jsonl(Q.DATA / "tuning" / "teacher_filtered.jsonl", kept_all)
    Q.write_jsonl(Q.DATA / "tuning" / "corrections_only.jsonl", [r for r in clean if r["source"] == "correction"])
    print("Day 2:", json.dumps(d21), json.dumps(d22), json.dumps(d24), "teacher set", len(kept_all), "wrong in it", escaped_all)


def test_outputs(b: dict) -> None:
    """The larger prompted model on the held-out test set, for Lab 4.1."""
    outs = outputs(b["test"], prompts.FROZEN, "large", examples())
    Q.write_jsonl(REF / "outputs" / "large_test.jsonl", [{"id": r["id"], "output": o} for r, o in zip(b["test"], outs)])
    print("larger prompted model on the test set:", sum(C.same_call(C.parse_call(o), r["gold"]) for r, o in zip(b["test"], outs)), "of", len(outs))


if __name__ == "__main__":
    built = P.build()
    day1()
    day2(built)
    test_outputs(built)
