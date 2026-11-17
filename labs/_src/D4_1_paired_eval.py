# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 4.1 · Three systems, two languages
#
# Day 4 · Session 1 (Paired evaluation against the base) · 25 min.
#
# The test set is opened today, once. Three systems answered the same 400 held-out requests: the base
# model as it was, the tuned model (the 4-bit base with the adapter of Lab 3.2), and the larger model
# with the frozen instructions of Day 1. The same requests for all three: that is what makes the
# comparison **paired**, and a paired comparison sees differences that two separate scores hide.
#
# **Reference state.** `data/requests_test.jsonl` and the three sets of outputs in
# `data/reference/outputs/`: `base_test.jsonl`, `tuned_test.jsonl` (both produced by the course model on
# a processor) and `large_test.jsonl` (the larger model; a scripted stand-in in the course's own run,
# see the lab guide). No model runs in this lab: the outputs are data. If you trained on the hub tier,
# the stretch of the lab guide shows how to write your own outputs in this format.
#
# **Tasks**
# - [ ] 1 · Grade the three sets of outputs (4')
# - [ ] 2 · Paired differences with a bootstrap interval (8')
# - [ ] 3 · The same comparison by language (6')
# - [ ] 4 · Write the comparison table and its reading (7')
# - [ ] ✱ Stretch · Test the tuned model against the larger one with an exact test on the discordant pairs

# %%
#@setup
Q.banner("Lab 4.1")

# %%
import math
import random

test = Q.load("requests_test")
outputs = {name: {r["id"]: r["output"] for r in Q.reference(f"outputs/{name}_test")} for name in ("base", "tuned", "large")}
print(len(test), "test requests ·", sum(r["lang"] == "ar" for r in test), "Arabic,", sum(r["lang"] == "en" for r in test), "English")

# %% [markdown]
# ## 1 · Grade the three sets of outputs (4')
#
# ✏️ **Exercise** · Complete `correct_list`: for one system, return a list with `1` where its output is
# the right call for the request and `0` where it is not, in the order of `test`. Use `C.parse_call` and
# `C.same_call`.

# %%
def correct_list(system):
    # BEGIN SOLUTION
    return [int(C.same_call(C.parse_call(outputs[system][r["id"]]), r["gold"])) for r in test]
    # END SOLUTION

# %%
# Check, with a safety net
def _correct_list(system):
    return [int(C.same_call(C.parse_call(outputs[system][r["id"]]), r["gold"])) for r in test]

try:
    right = {name: correct_list(name) for name in outputs}
    assert right == {name: _correct_list(name) for name in outputs}
except NotImplementedError:
    right = {name: _correct_list(name) for name in outputs}
    print("correct_list is not written yet: using the reference grades")
for name, r in right.items():
    print(f"{name:<6} {sum(r):>3} of {len(r)}  {sum(r) / len(r):.1%}")

# %% [markdown]
# ## 2 · Paired differences with a bootstrap interval (8')
#
# For each request, the difference between two systems is +1 (only the first is right), -1 (only the
# second) or 0 (both right, or both wrong). The mean of those differences is the difference in
# accuracy. Its uncertainty comes from the requests where the two systems **disagree**, and a bootstrap
# measures it: draw 400 requests with replacement, many times, and look at the spread of the mean.
#
# ✏️ **Exercise** · Complete `paired_bootstrap`: return `(difference, low, high)`, the mean paired
# difference and the 2.5th and 97.5th percentiles of `draws` bootstrap means.

# %%
def paired_bootstrap(a, b, draws=2000, seed=310):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    # BEGIN SOLUTION
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return sum(d) / n, means[int(0.025 * draws)], means[int(0.975 * draws) - 1]
    # END SOLUTION

# %%
def _paired_bootstrap(a, b, draws=2000, seed=310):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return sum(d) / n, means[int(0.025 * draws)], means[int(0.975 * draws) - 1]

DRAWS = Q.pick(2000, 300)
try:
    diff, low, high = paired_bootstrap(right["tuned"], right["base"], DRAWS)
    assert low <= diff <= high
except NotImplementedError:
    paired_bootstrap = _paired_bootstrap
    print("paired_bootstrap is not written yet: using the reference version")


def compare(a, b, rows=None):
    idx = range(len(test)) if rows is None else rows
    x, y = [right[a][i] for i in idx], [right[b][i] for i in idx]
    diff, low, high = paired_bootstrap(x, y, DRAWS)
    only_a, only_b = sum(p > q for p, q in zip(x, y)), sum(q > p for p, q in zip(x, y))
    return {"a": a, "b": b, "n": len(x), "difference": round(diff, 4), "low": round(low, 4), "high": round(high, 4), "only_a": only_a, "only_b": only_b}


pairs = [compare("tuned", "base"), compare("tuned", "large")]
for p in pairs:
    print(f"{p['a']} minus {p['b']}: {p['difference']:+.1%}  [{p['low']:+.1%}, {p['high']:+.1%}]  · only {p['a']} right: {p['only_a']} · only {p['b']} right: {p['only_b']}")

# %% [markdown]
# 👀 **Look at** the two right-hand counts. Both systems can be right on most requests and still
# disagree on dozens. The interval is built from those disagreements, which is why it is narrower than
# the two unpaired intervals of Lab 1.2 would suggest. An interval that does not contain zero supports
# "these two differ on requests like these"; one that does contain zero supports nothing either way.

# %% [markdown]
# ## 3 · The same comparison by language (6')
#
# An overall score can hide a slice. The chamber's members write in Arabic and in English, and a model
# that is better overall and worse in Arabic is not better for the chamber.

# %%
by_lang = {}
for lang in ("ar", "en"):
    rows = [i for i, r in enumerate(test) if r["lang"] == lang]
    by_lang[lang] = {"n": len(rows), **{name: sum(right[name][i] for i in rows) for name in right},
                     "tuned_vs_base": compare("tuned", "base", rows), "tuned_vs_large": compare("tuned", "large", rows)}
    b = by_lang[lang]
    print(f"{lang}: base {b['base']} · tuned {b['tuned']} · large {b['large']} of {b['n']}")
    for key in ("tuned_vs_base", "tuned_vs_large"):
        p = b[key]
        print(f"     {p['a']} minus {p['b']}: {p['difference']:+.1%}  [{p['low']:+.1%}, {p['high']:+.1%}]")

# %% [markdown]
# ## 4 · Write the comparison table and its reading (7')
#
# ✏️ **Exercise** · Complete `reading`: return one sentence for a comparison: "X is better than Y",
# "Y is better than X", or "no difference shown between X and Y", decided by whether the interval
# excludes zero.

# %%
def reading(p):
    # BEGIN SOLUTION
    if p["low"] > 0:
        return f"{p['a']} is better than {p['b']}"
    if p["high"] < 0:
        return f"{p['b']} is better than {p['a']}"
    return f"no difference shown between {p['a']} and {p['b']}"
    # END SOLUTION

# %%
def _reading(p):
    return f"{p['a']} is better than {p['b']}" if p["low"] > 0 else f"{p['b']} is better than {p['a']}" if p["high"] < 0 else f"no difference shown between {p['a']} and {p['b']}"

try:
    assert reading({"a": "x", "b": "y", "low": -0.01, "high": 0.02}).startswith("no difference")
except NotImplementedError:
    reading = _reading
    print("reading is not written yet: using the reference version")
Q.show("Held-out test set, 400 requests", [(name, f"{sum(r)} of {len(r)} ({sum(r) / len(r):.1%})") for name, r in right.items()])
lines = []
for scope, items in (("all", pairs), ("Arabic", [by_lang["ar"]["tuned_vs_base"], by_lang["ar"]["tuned_vs_large"]]), ("English", [by_lang["en"]["tuned_vs_base"], by_lang["en"]["tuned_vs_large"]])):
    for p in items:
        lines.append({"scope": scope, **p, "reading": reading(p)})
        print(f"{scope:<8} {p['difference']:+.1%} [{p['low']:+.1%}, {p['high']:+.1%}]  {reading(p)}")
Q.write_json(Q.artifacts("4.1") / "comparison.json", {"scores": {k: sum(v) for k, v in right.items()}, "of": len(test), "by_lang": {k: {x: v[x] for x in ("n", "base", "tuned", "large")} for k, v in by_lang.items()}, "comparisons": lines})

# %% [markdown]
# 👀 **Look at** what the table lets you say to Maysoon Al-Dabbagh, and what it does not. The tuned
# model against its base is the claim "tuning worked". The tuned model against the larger prompted model
# is the claim the decision rests on, and it is a different sentence. Write both in the memo, with their
# intervals, by language.

# %% [markdown]
# ## ✱ Stretch · An exact test on the discordant pairs
#
# McNemar's exact test asks: among the requests where the two systems disagree, is the split further
# from half and half than chance would give?

# %%
def exact_p(only_a, only_b):
    n, k = only_a + only_b, min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


for p in pairs:
    print(f"{p['a']} against {p['b']}: {p['only_a']} to {p['only_b']} on the discordant pairs · exact p = {exact_p(p['only_a'], p['only_b']):.4f}")

# %% [markdown]
# ## Check yourself
#
# 1. Two systems score 380 and 372 of 400. Can you say which is better from the two scores alone?
# 2. What makes a comparison paired, and what does pairing buy?
# 3. The tuned model is better overall and no better in Arabic. What goes in the memo?
#
# <details><summary>Answers</summary>
#
# 1. No. It depends on how many requests they disagree on: 8 against 0 is a clear difference, 40
#    against 32 is not.
# 2. The same requests answered by both systems. Request difficulty cancels, so the interval depends
#    only on the disagreements.
# 3. Both sentences, with both intervals. The chamber serves both languages; an overall figure alone
#    would hide the one it should know about.
#
# </details>
#
# **Carry forward.** `artifacts/4.1/comparison.json` feeds the model card and the memo of Lab 4.4.
