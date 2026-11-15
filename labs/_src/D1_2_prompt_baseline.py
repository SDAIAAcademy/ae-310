# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 1.2 · Freeze the baseline
#
# Day 1 · Session 2 (The baseline tuning must beat) · 25 min.
#
# A tuned model that beats a careless prompt has proved nothing. You improve the instructions one change at
# a time, score each version on the same 100 requests with an interval, and freeze the best one: its
# text, its hash, the set and the score. From now on every adapted model is compared with this baseline.
#
# **Reference state.** `data/requests_100.jsonl` and the parts of the instructions in `prompts.py`.
# Nothing from Lab 1.1.
#
# **Tasks**
# - [ ] 1 · Score version 0 with an interval (4')
# - [ ] 2 · Add schema hints (5')
# - [ ] 3 · Add the code table and the rules (6')
# - [ ] 4 · Add four examples (5')
# - [ ] 5 · Freeze the baseline and list what remains (5')
# - [ ] ✱ Stretch · Run the frozen instructions on the larger model: how much room is left above the baseline?

# %%
#@setup
Q.banner("Lab 1.2")

# %%
import hashlib
import math

import prompts

requests = Q.load("requests_100")
REF = Q.reference("d1_2_prompt_scores")


def score(system, examples=None, role="small"):
    """Right answers out of 100 for one version of the instructions."""
    results = Q.run(requests, system, role, examples)
    graded = Q.grade([r["text"] for r in results], requests)
    return sum(g["correct"] for g in graded), graded

# %% [markdown]
# ## 1 · Score version 0 with an interval (4')
#
# 100 requests are a sample. A score of 27 out of 100 could have been 20 or 35 on another hundred. The
# Wilson interval says how far: for `k` right out of `n`, with `z = 1.96` for 95%,
#
# `centre = (p + z²/2n) / (1 + z²/n)` and `half = z · sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)`, with `p = k/n`.
#
# ✏️ **Exercise** · Complete `wilson`.

# %%
def wilson(k, n, z=1.96):
    """The 95% Wilson interval for k right out of n, as (low, high)."""
    # BEGIN SOLUTION
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
    # END SOLUTION

# %%
# Check, with a safety net
try:
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 0.001 and abs(hi - 0.5962) < 0.001, (lo, hi)
except NotImplementedError:
    def wilson(k, n, z=1.96):  # the reference version, so the lab goes on
        p = k / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return centre - half, centre + half
    print("wilson is not written yet: using the reference version")

results = {}
k, _ = score(prompts.V0)
results["v0"] = k
print(f"version 0: {k} of 100, 95% interval {wilson(k, 100)[0]:.2f} to {wilson(k, 100)[1]:.2f}")

# %% [markdown]
# ## 2 · Add schema hints (5')
#
# Version 0 never told the model what a call looks like. `prompts.SCHEMA_HINT` does: the wrapper, the
# tool names with their argument keys, and how to write a number, a name and a city.
#
# 👀 **Look at** the text before you run it. Which of the format failures of Lab 1.1 does each sentence
# address?

# %%
print(prompts.SCHEMA_HINT)
print()
v1 = prompts.build(schema=True)
results["v1"], graded_v1 = score(v1)
print(f"version 1 (schema hints): {results['v1']} of 100, interval {wilson(results['v1'], 100)[0]:.2f} to {wilson(results['v1'], 100)[1]:.2f}")

# %% [markdown]
# ## 3 · Add the code table and the rules (6')
#
# The knowledge class needs the chamber's activity table **in the context**; the escalation failures need
# the rule stated. One change at a time, so that you know what each one bought.
#
# ✏️ **Exercise** · Build version 2 (schema hints and the code table) and version 3 (version 2 and the
# rules) with `prompts.build`, and score both.

# %%
def versions_2_and_3():
    """Return (v2, v3): the two instruction texts."""
    # BEGIN SOLUTION
    v2 = prompts.build(schema=True, table=True)
    v3 = prompts.build(schema=True, table=True, rules=True)
    return v2, v3
    # END SOLUTION

# %%
try:
    v2, v3 = versions_2_and_3()
    assert "BR-101" in v2 and "escalate_to_staff" in v3 and "Never invent" in v3 and "Never invent" not in v2
except NotImplementedError:
    v2, v3 = prompts.build(schema=True, table=True), prompts.FROZEN
    print("versions_2_and_3 is not written yet: using the reference versions")
results["v2"], _ = score(v2)
results["v3"], _ = score(v3)
for name in ("v2", "v3"):
    print(f"{name}: {results[name]} of 100, interval {wilson(results[name], 100)[0]:.2f} to {wilson(results[name], 100)[1]:.2f}")

# %% [markdown]
# ## 4 · Add four examples (5')
#
# Examples show what instructions can only describe. They come from the **validation** requests: an
# example taken from the 100 requests you score on would be answered from memory, and one taken from the
# test set would spoil every later comparison. No example carries a member's personal data.

# %%
val = {r["id"]: r for r in Q.load("requests_val")}
examples = [{"text": val[i]["text"], "call": val[i]["gold"]} for i in prompts.EXAMPLE_IDS]
for ex in examples:
    print(ex["text"], "->", ex["call"]["name"])
    assert not any(C.find_personal_data(ex["text"]).values())
results["v4"], graded_v4 = score(v3, examples)
print(f"\nversion 4 (version 3 and four examples): {results['v4']} of 100, interval {wilson(results['v4'], 100)[0]:.2f} to {wilson(results['v4'], 100)[1]:.2f}")
if Q.using_standin():
    assert results == {k: REF[k]["correct"] for k in results}, results

# %% [markdown]
# ## 5 · Freeze the baseline and list what remains (5')
#
# A baseline is three things that never change again: the instructions (identified by a hash), the set
# and the score with its interval. If any of the three moves, it is a new baseline.
#
# ✏️ **Exercise** · Complete `freeze`: return a dictionary with the keys `prompt_sha256` (of the
# instruction text followed by the example texts), `n`, `correct`, `accuracy` and `interval`.

# %%
def freeze(system, examples, correct, n):
    text = system + "".join(ex["text"] + C.render_call(ex["call"]) for ex in examples)
    # BEGIN SOLUTION
    lo, hi = wilson(correct, n)
    return {"prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "n": n, "correct": correct,
            "accuracy": correct / n, "interval": [round(lo, 3), round(hi, 3)]}
    # END SOLUTION

# %%
try:
    baseline = freeze(v3, examples, results["v4"], len(requests))
    assert len(baseline["prompt_sha256"]) == 64 and baseline["accuracy"] == results["v4"] / 100
except NotImplementedError:
    baseline = {"prompt_sha256": "write freeze to compute it", "n": 100, "correct": results["v4"], "accuracy": results["v4"] / 100,
                "interval": [round(x, 3) for x in wilson(results["v4"], 100)]}
    print("freeze is not written yet: the baseline below has no hash")
out = Q.artifacts("1.2") / "baseline.json"
Q.write_json(out, {**baseline, "set": "data/requests_100.jsonl", "model": "stand-in" if Q.using_standin() else Q.SMALL_MODEL, "versions": results})
print("frozen:", baseline)
print("saved", out.relative_to(Q.LABS))
print()
wrong = [(r, g) for r, g in zip(requests, graded_v4) if not g["correct"]]
print(f"What remains after the best instructions: {len(wrong)} failures. Five of them:")
for r, g in wrong[:5]:
    print(" ", r["text"][:90])
    print("     answered:", g["output"].replace("\n", " ")[:110])

# %% [markdown]
# 👀 **Look at** what remains. The schema is there, the table is there, the rules are written, and the
# model still takes the partner's number or renews a membership that it should have escalated. These are
# decisions, not missing information. That residue is the case for tuning, and the size of it (against
# the larger model in the stretch) is how much tuning could win.

# %% [markdown]
# ## ✱ Stretch · The larger model with the frozen instructions

# %%
k_large, _ = score(v3, examples, role="large")
print(f"larger model, frozen instructions: {k_large} of 100, interval {wilson(k_large, 100)[0]:.2f} to {wilson(k_large, 100)[1]:.2f}")
print(f"room above the baseline: {k_large - results['v4']} requests in 100")

# %% [markdown]
# ## Check yourself
#
# 1. Version 2 scored higher than version 1. Do their intervals overlap, and what does that allow you to say?
# 2. Why must the examples not come from the 100 requests?
# 3. The team later edits one sentence of the frozen instructions. What happens to the baseline?
#
# <details><summary>Answers</summary>
#
# 1. On 100 requests the intervals are about 20 points wide, so neighbouring versions overlap. The
#    versions were scored on the **same** requests, which is what a paired comparison uses (Day 4); the
#    unpaired intervals alone do not prove a difference.
# 2. The model would see four of the answers it is scored on. The score would rise without the
#    instructions being better.
# 3. The hash changes, so it is a new baseline and must be scored again. Comparing a tuned model with
#    the old score and the new text is comparing with nothing.
#
# </details>
#
# **Carry forward.** `artifacts/1.2/baseline.json`. The supplied copy of this baseline's instructions is
# `prompts.FROZEN` with `prompts.EXAMPLE_IDS`: Day 2 uses them for the traces and the teacher.
