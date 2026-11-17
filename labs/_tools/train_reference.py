"""Train the supplied adapters of Days 3 and 4 on the course model and record their results.

    uv run python _tools/train_reference.py [stage ...]

Stages: base r8 bad 4bit sweep corrections distilled replay day4. With no argument, all of them.
Run once by the course authors on a processor; adapters/ and data/reference/ are committed.
Every number the decks quote for the course model comes from the files this script writes.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

import chamber as C  # noqa: E402
import prepare_data as P  # noqa: E402
import qayd as Q  # noqa: E402
import tuning as U  # noqa: E402

REF = Q.REFERENCE
OUT = REF / "outputs"
ADAPTERS = Q.ADAPTERS
STEPS, LR, BATCH, RANK = int(os.environ.get("REF_STEPS", "300")), 2e-3, 8, 8  # REF_STEPS: a smoke test of this script
RANKS, RATES = [2, 8, 32], [2e-4, 2e-3, 2e-2]
HELD_OUT = (32, 2e-3)  # the setting Lab 3.3 leaves for the participant

TRAIN = U.rows_from(Q.load("tuning/train"))
VAL_ROWS = U.rows_from(Q.load("tuning/val"))
VAL = Q.load("requests_val")
TEST = Q.load("requests_test")


def score(model, tok, rows):
    return U.grade(U.generate(model, tok, [r["text"] for r in rows]), rows)


def by_slice(graded, rows):
    table = {}
    for g, r in zip(graded, rows):
        for s in U.slices(r):
            right, n = table.get(s, (0, 0))
            table[s] = (right + g["correct"], n + 1)
    return {s: {"right": a, "of": n} for s, (a, n) in sorted(table.items())}


def summary(graded):
    return {"right": sum(g["correct"] for g in graded), "valid": sum(g["valid"] for g in graded), "of": len(graded),
            "ar": round(U.accuracy(graded, "ar"), 4), "en": round(U.accuracy(graded, "en"), 4)}


def fit(rows, bits=16, rank=RANK, lr=LR, steps=STEPS, schedule="cosine", warmup=0.06, name=None, evaluate=True):
    base, tok = U.load_base("tiny", bits)
    model = U.add_adapter(base, rank=rank)
    log = U.train(model, tok, rows, VAL_ROWS, steps=steps, lr=lr, batch=BATCH, schedule=schedule, warmup=warmup)
    log.update({"rank": rank, "bits": bits, "base_memory_mb": round(U.base_memory_mb(base), 3)})
    graded = None
    if evaluate:
        graded = score(model, tok, VAL)
        log["val"] = summary(graded)
        log["val_slices"] = by_slice(graded, VAL)
    if name:
        U.save_adapter(model, ADAPTERS / name, log)
    print(name or f"r{rank} lr{lr}", "·", log["seconds"], "s ·", log["curve"][-1], "·", log.get("val"), flush=True)
    return model, tok, log, graded


def stage_base():
    result = {}
    for bits in (16, 4):
        base, tok = U.load_base("tiny", bits)
        graded = score(base, tok, VAL)
        result[str(bits)] = {"memory_mb": round(U.base_memory_mb(base), 3), "val": summary(graded), "val_slices": by_slice(graded, VAL),
                             "val_loss": round(U.loss_on(base, U.encode(tok, VAL_ROWS, which="tiny")), 4)}
        if bits == 16:
            Q.write_jsonl(OUT / "base_val.jsonl", [{"id": g["id"], "output": g["output"]} for g in graded])
            # the base answers in its older text format: how often is the content right all the same?
            needs_call = [(g, r) for g, r in zip(graded, VAL) if r["gold"]["name"] != "escalate_to_staff"]
            old = lambda gold: "CALL " + gold["name"] + " " + " ".join(f"{k}={v}" for k, v in gold["arguments"].items())
            result["legacy"] = {"right": sum(g["output"].strip() == old(r["gold"]) for g, r in needs_call), "of": len(needs_call)}
        print("base", bits, result[str(bits)], flush=True)
    result["parameters"] = sum(p.numel() for p in base.parameters())
    Q.write_json(REF / "d3_base.json", result)


def stage_r8():
    _, _, log, graded = fit(TRAIN, name="registry-16bit")
    Q.write_json(REF / "d3_1_run.json", log)
    Q.write_jsonl(OUT / "r8_val.jsonl", [{"id": g["id"], "output": g["output"]} for g in graded])


def stage_bad():
    # a learning rate fifty times too high, no warm-up, no decay: the run that went wrong
    _, _, log, _ = fit(TRAIN, lr=0.1, schedule="constant", warmup=0.0, steps=150)
    Q.write_json(REF / "d3_1_bad_run.json", log)


def stage_4bit():
    _, _, log, _ = fit(TRAIN, bits=4, name="registry-4bit")
    Q.write_json(REF / "d3_2_4bit.json", log)


def stage_sweep():
    keep = ("rank", "lr", "trainable", "seconds", "val", "curve")
    runs, held = [], None
    first = Q.read_json(REF / "d3_1_run.json")
    for rank in RANKS:
        for lr in RATES:
            if (rank, lr) == (RANK, LR):
                log = first
            else:
                _, _, log, _ = fit(TRAIN, rank=rank, lr=lr)
            row = {k: log[k] for k in keep}
            if (rank, lr) == HELD_OUT:
                held = row
            else:
                runs.append(row)
    Q.write_json(REF / "d3_3_sweep.json", {"steps": STEPS, "batch": BATCH, "runs": runs, "held_out": held})


def stage_corrections():
    rows = U.rows_from(Q.load("tuning/corrections_only"))
    _, _, log, _ = fit(rows, name="corrections-only")
    Q.write_json(REF / "d3_4_corrections.json", log)


def teacher_errors():
    """The answer key for the teacher's kept labels: which are wrong, and what the right call is."""
    index = Q.read_json(Q.DATA / "standin" / "index.json")  # keyed by the request text, so that repeated texts share an entry
    key = {p["id"]: index[P.stable_hash(p["text"])]["gold"] for p in Q.load("pool_unlabelled")}
    wrong = []
    for e in Q.load("tuning/teacher_filtered"):
        label = e["messages"][1]["tool_calls"][0]["function"]
        if not C.same_call(label, key[e["id"]]):
            wrong.append({"id": e["id"], "lang": e["lang"], "text": e["messages"][0]["content"], "teacher": label, "gold": key[e["id"]]})
    Q.write_jsonl(REF / "d3_4_teacher_errors.jsonl", wrong)
    return wrong


def stage_distilled():
    rows = U.rows_from(Q.load("tuning/teacher_filtered"))
    model, tok, log, graded = fit(rows, name="distilled", steps=min(200, STEPS))
    wrong = teacher_errors()
    outs = U.generate(model, tok, [w["text"] for w in wrong])
    log["teacher_errors"] = {"in_set": len(wrong), "copied": sum(C.same_call(C.parse_call(o), w["teacher"]) for o, w in zip(outs, wrong)),
                             "right": sum(C.same_call(C.parse_call(o), w["gold"]) for o, w in zip(outs, wrong))}
    print(log["teacher_errors"], flush=True)
    U.save_adapter(model, ADAPTERS / "distilled", log)
    Q.write_json(REF / "d3_4_distilled.json", log)
    Q.write_jsonl(OUT / "distilled_val.jsonl", [{"id": g["id"], "output": g["output"]} for g in graded])


def replay_rows():
    """200 general and refusal examples that are not in the two check files: what replay mixes into the tuning set."""
    rng = random.Random(777)
    seen = {r["text"] for name in ("general_check", "refusal_check") for r in Q.load(name)}
    companies = P.make_registry(rng, 60)
    rows = []
    while len(rows) < 150:
        g = P.general_item(rng)
        if g["text"] not in seen:
            seen.add(g["text"])
            rows.append(g)
    while len(rows) < 200:
        r = P.refusal_item(rng, companies)
        if r["text"] not in seen:
            seen.add(r["text"])
            rows.append(r)
    examples = [{"id": f"P-{i + 1:03d}", "source": "replay", "lang": r.get("lang", "en"),
                 "messages": [{"role": "user", "content": r["text"]}, {"role": "assistant", "content": r["answer"]}]} for i, r in enumerate(rows)]
    Q.write_jsonl(Q.DATA / "tuning" / "replay.jsonl", examples)
    return examples


def stage_replay():
    rows = TRAIN + U.rows_from(replay_rows())
    _, _, log, _ = fit(rows, bits=4, name="registry-replay")
    Q.write_json(REF / "d4_2_replay_run.json", log)


def checks(model, tok):
    general, refusal = Q.load("general_check"), Q.load("refusal_check")
    g = U.grade_text(U.generate(model, tok, [r["text"] for r in general]), general)
    f = U.generate(model, tok, [r["text"] for r in refusal])
    tasks = {}
    for row, item in zip(g, general):
        right, n = tasks.get(item["task"], (0, 0))
        tasks[item["task"]] = (right + row["correct"], n + 1)
    return {"general": {"right": sum(x["correct"] for x in g), "of": len(g), "by_task": {t: {"right": a, "of": n} for t, (a, n) in sorted(tasks.items())}},
            "refusal": {"refused": sum(U.refused(o) for o in f), "of": len(f)},
            "outputs": {"general": [x["output"] for x in g], "refusal": f}}


def stage_day4():
    """Test-set outputs of the base and the tuned model, and the regression checks of three systems."""
    base, tok = U.load_base("tiny", 4)
    graded = score(base, tok, TEST)
    Q.write_jsonl(OUT / "base_test.jsonl", [{"id": g["id"], "output": g["output"]} for g in graded])
    result = {"base": {"test": summary(graded), **checks(base, tok)}}
    for key, name in (("tuned", "registry-4bit"), ("replay", "registry-replay")):
        base, tok = U.load_base("tiny", 4)
        model = U.load_adapter(base, ADAPTERS / name)
        graded = score(model, tok, TEST)
        Q.write_jsonl(OUT / f"{key}_test.jsonl", [{"id": g["id"], "output": g["output"]} for g in graded])
        result[key] = {"adapter": name, "test": summary(graded), "test_slices": by_slice(graded, TEST), **checks(model, tok)}
    large = {r["id"]: r for r in Q.read_jsonl(OUT / "large_test.jsonl")}
    lg = U.grade([large[r["id"]].get("output", large[r["id"]].get("text", "")) for r in TEST], TEST)
    result["large"] = {"test": summary(lg)}
    for key in result:
        print(key, {k: v for k, v in result[key].items() if k != "outputs"}, flush=True)
    Q.write_json(REF / "d4_checks.json", result)


STAGES = {"base": stage_base, "r8": stage_r8, "bad": stage_bad, "4bit": stage_4bit, "sweep": stage_sweep,
          "corrections": stage_corrections, "distilled": stage_distilled, "replay": stage_replay, "day4": stage_day4}

if __name__ == "__main__":
    for stage in sys.argv[1:] or list(STAGES):
        print("==", stage, flush=True)
        STAGES[stage]()
