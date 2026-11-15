# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 2.1 · Traces to examples
#
# Day 2 · Session 1 (Tuning data from traces and corrections) · 25 min.
#
# For a month Qayd ran as a pilot with the frozen instructions of Day 1. Every request left a trace: the
# request, Qayd's answer and what the registry desk did with it. Khuzama Al-Rumaizan's team accepted most
# answers, corrected some and gave up on others. You turn those 500 traces into tuning examples in chat
# format, and you count exactly what you kept and what you dropped, because the dataset card will ask.
#
# **Reference state.** `data/traces_500.jsonl` (500 traces) and `data/corrections.jsonl` (the desk's
# corrections). Nothing from Day 1.
#
# **Tasks**
# - [ ] 1 · Read the traces and their statuses (4')
# - [ ] 2 · Join the corrections to the traces (5')
# - [ ] 3 · Validate every label against the tool interface (7')
# - [ ] 4 · Write chat-format examples (5')
# - [ ] 5 · Count what was kept and dropped (4')
# - [ ] ✱ Stretch · Find accepted traces whose call disagrees with the request text

# %%
#@setup
Q.banner("Lab 2.1")

# %%
import json
import re
from collections import Counter

traces = Q.load("traces_500")
corrections = Q.load("corrections")
REF = Q.reference("d2_1_examples")
print(len(traces), "traces,", len(corrections), "corrections")

# %% [markdown]
# ## 1 · Read the traces and their statuses (4')
#
# A trace has a status:
#
# - `accepted`: the desk let Qayd's call go through. The label is Qayd's own output.
# - `corrected`: the desk replaced the call. The label is in `corrections.jsonl`.
# - `abandoned`: the desk handled the request by hand and recorded nothing. There is **no label**.
#
# 👀 **Look at** one trace of each status. Which of the three can teach the model something it does not
# already do?

# %%
print(Counter(t["status"] for t in traces))
for status in ("accepted", "corrected", "abandoned"):
    t = next(t for t in traces if t["status"] == status)
    print(f"\n[{status}] {t['id']}  {t['text'][:110]}")
    print("   Qayd answered:", t["output"].replace("\n", " ")[:120])

# %% [markdown]
# ## 2 · Join the corrections to the traces (5')
#
# ✏️ **Exercise** · Complete `label_of`: return the tool call that labels a trace, or `None` when it has
# none. For an accepted trace it is the call parsed from Qayd's output (`C.parse_call`); for a corrected
# trace it is the desk's call; an abandoned trace has no label.

# %%
correction_of = {c["trace_id"]: c["call"] for c in corrections}


def label_of(trace):
    # BEGIN SOLUTION
    if trace["status"] == "accepted":
        return C.parse_call(trace["output"])
    if trace["status"] == "corrected":
        return correction_of.get(trace["id"])
    return None
    # END SOLUTION

# %%
# Check, with a safety net
def _label_of(trace):
    if trace["status"] == "accepted":
        return C.parse_call(trace["output"])
    return correction_of.get(trace["id"]) if trace["status"] == "corrected" else None

try:
    labels = {t["id"]: label_of(t) for t in traces}
    assert labels == {t["id"]: _label_of(t) for t in traces}, "an accepted trace is labelled by its own output, a corrected one by the desk"
except NotImplementedError:
    labels = {t["id"]: _label_of(t) for t in traces}
    print("label_of is not written yet: using the reference labels")
print("traces with a label:", sum(v is not None for v in labels.values()), "· without:", sum(v is None for v in labels.values()))

# %% [markdown]
# ## 3 · Validate every label against the tool interface (7')
#
# A correction is typed by a person at a busy desk. A label is only as good as the tool it calls would
# accept: `C.schema_errors(call)` lists why the registry would reject a call (an unknown tool, a missing
# argument, a number that is not 10 Western digits, a city outside the list).
#
# ✏️ **Exercise** · Complete `validate`: split the labelled traces into those whose label the interface
# accepts and those it rejects. Return `(kept, rejected)`, where `kept` is a list of `(trace, call)` and
# `rejected` a list of `(trace, call, errors)`.

# %%
def validate(traces, labels):
    kept, rejected = [], []
    # BEGIN SOLUTION
    for t in traces:
        call = labels[t["id"]]
        if call is None:
            continue
        errors = C.schema_errors(call)
        if errors:
            rejected.append((t, call, errors))
        else:
            kept.append((t, call))
    # END SOLUTION
    return kept, rejected

# %%
try:
    kept, rejected = validate(traces, labels)
    if not kept:
        raise NotImplementedError
    assert all(not C.schema_errors(c) for _, c in kept)
except NotImplementedError:
    kept = [(t, labels[t["id"]]) for t in traces if labels[t["id"]] is not None and not C.schema_errors(labels[t["id"]])]
    rejected = [(t, labels[t["id"]], C.schema_errors(labels[t["id"]])) for t in traces if labels[t["id"]] is not None and C.schema_errors(labels[t["id"]])]
    print("validate is not written yet: using the reference split")
print(len(kept), "labels kept,", len(rejected), "rejected by the interface")
for t, call, errors in rejected[:5]:
    print(f"  {t['id']} [{t['status']}] {json.dumps(call['arguments'], ensure_ascii=False)[:80]}  ->  {errors[0]}")
assert len(kept) == REF["examples"] and len(rejected) == REF["dropped"]["correction rejected by the interface"], (len(kept), len(rejected))

# %% [markdown]
# 👀 **Look at** the rejected labels. Every one is a correction: a registration number one digit short, a
# city typed in Arabic, digits left in Arabic-Indic form. The desk knew what it meant; the registry would
# not. A model trained on these would learn to make the same mistakes with confidence.

# %% [markdown]
# ## 4 · Write chat-format examples (5')
#
# A tuning example is a short conversation: the member's request, then the assistant's turn holding the
# tool call. This is the structure a chat template renders (Lab 2.3), and it is the same structure the
# server will receive in production:
#
# ```json
# {"id": "R-0001", "source": "trace", "lang": "en",
#  "messages": [{"role": "user", "content": "..."},
#               {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]}]}
# ```
#
# ✏️ **Exercise** · Complete `to_example`. `source` is `"trace"` for an accepted trace and
# `"correction"` for a corrected one.

# %%
def to_example(trace, call):
    # BEGIN SOLUTION
    return {"id": trace["id"], "source": "trace" if trace["status"] == "accepted" else "correction", "lang": trace["lang"],
            "messages": [{"role": "user", "content": trace["text"]},
                         {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]}]}
    # END SOLUTION

# %%
try:
    examples = [to_example(t, c) for t, c in kept]
    assert examples[0]["messages"][0]["role"] == "user" and examples[0]["messages"][1]["tool_calls"][0]["function"]["name"] in C.TOOL_NAMES
except NotImplementedError:
    examples = Q.reference("examples_traces")
    print("to_example is not written yet: using the supplied examples")
assert examples == Q.reference("examples_traces"), "the examples differ from the supplied copy"
print(json.dumps(examples[0], ensure_ascii=False, indent=1))

# %% [markdown]
# ## 5 · Count what was kept and dropped (4')
#
# The count is not bookkeeping. It is the first paragraph of the dataset card: where the examples came
# from, what was left out and why.

# %%
by_source = Counter(e["source"] for e in examples)
dropped = {"abandoned: no label": sum(t["status"] == "abandoned" for t in traces), "correction rejected by the interface": len(rejected)}
Q.show("From 500 traces", [("kept from accepted traces", by_source["trace"]), ("kept from corrections", by_source["correction"]),
                           *dropped.items(), ("examples", len(examples))])
assert by_source["trace"] + by_source["correction"] + sum(dropped.values()) == len(traces)
assert dict(by_source) == REF["by_source"]
out = Q.artifacts("2.1")
Q.write_jsonl(out / "examples_traces.jsonl", examples)
Q.write_json(out / "counts.json", {"by_source": dict(by_source), "dropped": dropped})
print("saved", (out / "examples_traces.jsonl").relative_to(Q.LABS))

# %% [markdown]
# 👀 **Look at** the proportions. Three quarters of the examples are calls Qayd already made correctly.
# They teach the format, and they teach nothing about the failures. The corrections, a quarter, are
# where the behaviour you want to change lives. Keep that in mind on Day 3, when one adapter is trained
# on corrections alone.

# %% [markdown]
# ## ✱ Stretch · Accepted, and wrong
#
# "Accepted" means nobody objected, not that the call was right. A rule check can catch some of these:
# a registration number in the call that does not appear in the request.

# %%
CR = re.compile(r"(?<![0-9])40[0-9]{8}(?![0-9])")
suspects = []
for e in examples:
    if e["source"] == "trace":
        args = e["messages"][1]["tool_calls"][0]["function"]["arguments"]
        numbers = CR.findall(C.western_digits(e["messages"][0]["content"]))
        if "cr_number" in args and args["cr_number"] not in numbers:
            suspects.append(e["id"])
print(len(suspects), "accepted traces call the registry with a number the request does not contain:", suspects)
print("The answer key says", REF["wrong_labels_left"], "accepted traces carry a wrong call. The rest pass this rule: they are wrong in ways only a person, or the test set, can see.")

# %% [markdown]
# ## Check yourself
#
# 1. Why is an abandoned trace not simply labelled with the correct call by you, now?
# 2. The interface rejected 12 corrections. Why not repair them by hand?
# 3. Which source of examples is most likely to contain wrong labels that pass every check?
#
# <details><summary>Answers</summary>
#
# 1. You could, and then it would be a new label made by you, with its own source name and its own
#    review. What you must not do is count it as a trace: the card has to say where every label came from.
# 2. A repair is a guess at what the desk meant. Twelve examples are not worth a guess that enters the
#    training set silently. Send them back to the desk, or drop them and say so.
# 3. Accepted traces. Nobody looked hard at them: acceptance is the absence of a complaint.
#
# </details>
#
# **Carry forward.** `artifacts/2.1/examples_traces.jsonl`. Lab 2.4 starts from the supplied
# `data/reference/examples_raw.jsonl`, which holds these examples and the teacher's.
