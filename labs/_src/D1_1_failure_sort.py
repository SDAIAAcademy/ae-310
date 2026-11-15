# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 1.1 · Sort the failures
#
# Day 1 · Session 1 (Prompt, retrieve, tune or distil) · 25 min.
#
# Qayd, the registration assistant of Al-Bayraq Chamber of Commerce, runs on a small open-weight model and
# fails too often. Before anyone trains anything, you find out **how** it fails. You run the small model
# and a larger reference model on the same 100 requests, grade every answer by code, sort each failure
# into one of three classes and fill the decision table: the evidence for outcome 1.
#
# **Assistants are off in this lab.** The failure analysis is yours: no AI assistant writes the sorting
# rule or reads the failures for you.
#
# **Reference state.** `data/requests_100.jsonl`: 100 requests, 50 in English and 50 in Arabic, each with
# the correct tool call. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Run both models on 100 requests (4')
# - [ ] 2 · Grade by code and read ten failures (6')
# - [ ] 3 · Write the rule that sorts a failure (8')
# - [ ] 4 · Fill the decision table (7')
# - [ ] ✱ Stretch · Split the behaviour failures by language: is one language carrying them?

# %%
#@setup
Q.banner("Lab 1.1")

# %%
import json
from collections import Counter

import prompts

requests = Q.load("requests_100")
print(len(requests), "requests;", Counter(r["lang"] for r in requests))
print()
print("The instructions the pilot started with (version 0):")
print(prompts.V0)
print()
print("One request and its correct call:")
print(requests[8]["text"])
print(C.render_call(requests[8]["gold"]))

# %% [markdown]
# ## 1 · Run both models on 100 requests (4')
#
# `Q.run(rows, system, role)` sends every request with the same instructions to the small model
# (`role="small"`) or the larger one (`role="large"`). Answers are cached, so running the cell again is
# free. Without a key the scripted stand-in answers, and the first line of the lab said so.

# %%
small = Q.run(requests, prompts.V0, "small")
large = Q.run(requests, prompts.V0, "large")
print("answered by:", Counter(r["source"] for r in small + large))
print()
for i in (0, 1, 2):
    print(requests[i]["text"])
    print("  small :", small[i]["text"].replace("\n", " "))
    print("  larger:", large[i]["text"].replace("\n", " "))

# %% [markdown]
# ## 2 · Grade by code and read ten failures (6')
#
# An answer is **right** when it is exactly the correct tool call: the same tool, the same arguments.
# `C.parse_call` is strict on purpose: one JSON object with `name` and `arguments`, inside `<tool_call>`
# tags. The registry service will be just as strict.
#
# 👀 **Look at** the ten failures printed below before you write any rule. For each one, say in a word
# what went wrong. You will need those words in task 3.

# %%
graded_small = Q.grade([r["text"] for r in small], requests)
graded_large = Q.grade([r["text"] for r in large], requests)
print(f"small model : {sum(g['correct'] for g in graded_small)} of 100 right, {sum(g['valid'] for g in graded_small)} valid calls")
print(f"larger model: {sum(g['correct'] for g in graded_large)} of 100 right, {sum(g['valid'] for g in graded_large)} valid calls")
print()
shown = 0
for row, g in zip(requests, graded_small):
    if not g["correct"] and shown < 10:
        shown += 1
        print(f"[{row['id']}] {row['text']}")
        print("   expected:", json.dumps(row["gold"], ensure_ascii=False))
        print("   answered:", g["output"].replace("\n", " "))
        print("   interface says:", C.schema_errors(g["call"]) or "fits")

# %% [markdown]
# ## 3 · Write the rule that sorts a failure (8')
#
# Three classes, because each one is fixed by a different lever:
#
# | Class | What it looks like | Why it matters |
# |---|---|---|
# | **knowledge** | The call is right except for the activity code: the chamber's own table, which no model was trained on | A model cannot be prompted or trained into knowing a table it has never seen and that changes every year |
# | **format** | No call that parses, or a call the interface rejects (wrong keys, digits not normalised, a code fence) | The model knows what to do and writes it wrongly |
# | **behaviour** | A well-formed call that fits the interface and is the wrong decision: the wrong tool, the wrong number, no escalation | The model decides wrongly |
#
# ✏️ **Exercise** · Complete `failure_class`. Return `None` for a right answer, otherwise one of the three
# class names. Decide in this order: no call that parses is `format`; a call that becomes right when you
# replace only its activity code by the correct one is `knowledge`; a call the interface rejects is
# `format`; everything else is `behaviour`.

# %%
def activity_key(gold):
    """The argument that holds an activity code in this call, or None."""
    if gold["name"] == "create_membership":
        return "activity_code"
    if gold["name"] == "update_record" and gold["arguments"].get("field") == "activity_code":
        return "value"
    return None


def failure_class(row, output):
    call = C.parse_call(output)
    gold = row["gold"]
    if C.same_call(call, gold):
        return None
    # BEGIN SOLUTION
    if call is None:
        return "format"
    key = activity_key(gold)
    if key and call["name"] == gold["name"] and key in call["arguments"]:
        patched = {"name": call["name"], "arguments": {**call["arguments"], key: gold["arguments"][key]}}
        if C.same_call(patched, gold):
            return "knowledge"
    if C.schema_errors(call):
        return "format"
    return "behaviour"
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
REF = Q.reference("d1_1_decision_table")
try:
    classes_small = Counter(failure_class(r, o["text"]) for r, o in zip(requests, small))
    classes_large = Counter(failure_class(r, o["text"]) for r, o in zip(requests, large))
    mine = True
except NotImplementedError:
    classes_small = Counter({None if k == "correct" else k: v for k, v in REF["counts"]["small"].items()})
    classes_large = Counter({None if k == "correct" else k: v for k, v in REF["counts"]["large"].items()})
    mine = False
    print("failure_class is not written yet: using the supplied counts")
probe = {"id": "x", "gold": C.call("renew_membership", cr_number="4011111111", years=2)}
if mine:
    assert failure_class(probe, "Sure, I renewed it.") == "format"
    assert failure_class(probe, C.render_call(C.call("renew_membership", cr_number="4011111111", years=1))) == "behaviour"
    assert failure_class(probe, C.render_call(probe["gold"])) is None
    if Q.using_standin():  # the supplied counts were recorded with the stand-in
        assert {("correct" if k is None else k): v for k, v in classes_small.items()} == REF["counts"]["small"], classes_small
print("small :", dict(classes_small))
print("larger:", dict(classes_large))

# %% [markdown]
# ## 4 · Fill the decision table (7')
#
# The table is the deliverable: counts per class for both models, and the approach you choose for each
# class. The larger model is your instrument here. A class the larger model **also** fails is not a
# matter of capability: more capability, by tuning or by a bigger model, will not fix it. A class the
# larger model gets right is a behaviour worth teaching the small one.
#
# ✏️ **Exercise** · Fill `approach` with one of `"prompting"`, `"retrieval"`, `"tuning"`,
# `"distillation"` for each class, and a reason of one sentence that quotes a count.

# %%
def choose_approach():
    # BEGIN SOLUTION
    return {
        "knowledge": ("retrieval", "The larger model fails it as often as the small one: the code table must be given, in context or by retrieval."),
        "format": ("prompting", "The larger model rarely fails it with the same instructions: try schema hints and examples first, tune for what remains."),
        "behaviour": ("distillation", "The larger model gets it right where the small one does not: teach the small model that behaviour, by tuning on corrections or on the larger model's calls."),
    }
    # END SOLUTION

# %%
try:
    approach = choose_approach()
    assert set(approach) == {"knowledge", "format", "behaviour"}, "one line per class"
except NotImplementedError:
    approach = {k: tuple(v) for k, v in REF["approach"].items()}
    print("choose_approach is not written yet: showing the supplied table")
table = [("class", "small model", "larger model", "approach", "why")]
for name in ("knowledge", "format", "behaviour"):
    table.append((name, classes_small[name], classes_large[name], approach[name][0], approach[name][1]))
table.append(("right", classes_small[None], classes_large[None], "", ""))
Q.show("Decision table · 100 requests · instructions version 0", table)
assert approach["knowledge"][0] in ("retrieval", "prompting"), "A table the model has never seen is not learnt by tuning: give it"
assert approach["behaviour"][0] in ("tuning", "distillation")
out = Q.artifacts("1.1") / "decision_table.json"
Q.write_json(out, {"small": {str(k): v for k, v in classes_small.items()}, "large": {str(k): v for k, v in classes_large.items()}, "approach": approach})
print("saved", out.relative_to(Q.LABS))

# %% [markdown]
# 👀 **Look at** the knowledge row. Sorting put format first, so a request with a broken format **and** a
# wrong code counts once, as format. That is why the larger model, which rarely breaks the format, shows
# more knowledge failures than the small one: the small model's are hidden behind its format failures.
# Fix the format and they will appear.

# %% [markdown]
# ## ✱ Stretch · Behaviour failures by language

# %%
by_lang = Counter(r["lang"] for r, o in zip(requests, small) if (failure_class(r, o["text"]) if mine else None) == "behaviour")
print("behaviour failures of the small model by language:", dict(by_lang) or "write failure_class first")

# %% [markdown]
# ## Check yourself
#
# 1. The larger model fails the knowledge class as often as the small one. What does that rule out?
# 2. Why is "the model answered in a code fence" a format failure and not a behaviour failure?
# 3. A colleague proposes tuning on the code table. What happens next January?
#
# <details><summary>Answers</summary>
#
# 1. That capability is the problem. A bigger or a tuned model still has not seen the chamber's table.
# 2. The decision inside the fence is right; only its wrapping is wrong. A parser, a schema hint or an
#    example fixes it without changing what the model decides.
# 3. The table is revised, the adapter still answers with last year's codes, and nobody notices until a
#    member is registered under the wrong activity. A table that changes belongs in context.
#
# </details>
#
# **Carry forward.** `artifacts/1.1/decision_table.json` is the evidence for outcome 1 and the first
# section of the decision memo of Lab 4.4.
