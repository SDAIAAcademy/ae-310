# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 4.4 · Cards, memo and capstone task
#
# Day 4 · Session 4 (Cards and the decision memo) · 25 min.
#
# Maysoon Al-Dabbagh decides on Thursday whether Qayd's tuned model goes into the registration service.
# She will read one page. Behind that page stand two documents that let someone else check it: the
# dataset card and the model card. You fill all three from the course's measured results, check every
# number in them against its source, and then choose the task of your own capstone.
#
# **Reference state.** The reports of Days 1 to 4 in `data/reference/` (your own from `artifacts/`
# where they exist), the three templates in `templates/`, and `data/onprem.json`. No model runs.
#
# **Tasks**
# - [ ] 1 · Fill the dataset card from the Day 2 reports (6')
# - [ ] 2 · Fill the model card from the evaluation (6')
# - [ ] 3 · Write the decision memo (8')
# - [ ] 4 · Choose the capstone task and its baseline (5')
# - [ ] ✱ Stretch · Exchange memos with a neighbour and challenge the weakest number

# %%
#@setup
Q.banner("Lab 4.4")

# %%
import re

TEMPLATES = Q.LABS / "templates"
out = Q.artifacts("4.4")
traces = Q.reference("d2_1_examples")
teacher = Q.reference("d2_2_teacher")
clean = Q.own_or_supplied("2.4", "overlap_report.json", "d2_4_clean.json")
run = Q.reference("d3_2_4bit")
checks = Q.reference("d4_checks")
comparison = Q.own_or_supplied("4.1", "comparison.json", "d4_1_comparison.json")
load = Q.own_or_supplied("4.3", "load.json", "d4_3_load.json")
cost = Q.own_or_supplied("1.4", "cost_table.json", "d1_4_cost_table.json")


KINDS = {"phone": "mobile numbers", "national_id": "identity numbers", "email": "e-mail addresses", "name": "names"}
hosted = cost["hosted_sar_per_thousand"]
hosted = hosted if isinstance(hosted, (int, float)) else next(iter(hosted.values()))  # the larger reference model is listed first


def fill(template, facts):
    """Put facts into a template's {placeholders}; a placeholder with no fact stays visible."""
    text = (TEMPLATES / template).read_text(encoding="utf-8")
    return re.sub(r"\{(\w+)\}", lambda m: str(facts.get(m.group(1), "{" + m.group(1) + " ?}")), text)

# %% [markdown]
# ## 1 · Fill the dataset card from the Day 2 reports (6')
#
# A dataset card answers: where did the examples come from, what was removed and by which rule, what is
# in the set now, and what is known to be wrong with it.
#
# ✏️ **Exercise** · Complete `dataset_facts`: the three entries marked ✏️. Each is one number read from
# a report; none is typed by hand.

# %%
def dataset_facts():
    facts = {
        "rows": clean["rows_out"], "from_traces": clean["by_source"]["trace"], "from_corrections": clean["by_source"]["correction"],
        "from_teacher": clean["by_source"]["teacher"], "arabic": clean["by_lang"]["ar"], "english": clean["by_lang"]["en"],
        "abandoned": traces["dropped"]["abandoned: no label"], "corrections_rejected": traces["dropped"]["correction rejected by the interface"],
        "teacher_yield": f"{teacher['yield']:.1%}", "audit_n": 30, "audit_wrong": teacher["audit_wrong_of_30"], "audit_bound": "11%",
        "personal_data": ", ".join(f"{v} {KINDS.get(k, k)}" for k, v in clean["personal_data_masked" if "personal_data_masked" in clean else "personal_data_found"].items()),
    }
    # BEGIN SOLUTION
    facts["near_duplicates"] = clean["near_duplicates_removed"]
    facts["leaks"] = clean["leaks_removed"]
    facts["collisions"] = clean["template_collisions_kept"]
    # END SOLUTION
    return facts

# %%
# Check, with a safety net
def _three():
    return {"near_duplicates": clean["near_duplicates_removed"], "leaks": clean["leaks_removed"], "collisions": clean["template_collisions_kept"]}

try:
    d_facts = dataset_facts()
    assert {k: d_facts.get(k) for k in _three()} == _three(), "read the three numbers from the overlap report"
except NotImplementedError:
    d_facts = {
        "rows": clean["rows_out"], "from_traces": clean["by_source"]["trace"], "from_corrections": clean["by_source"]["correction"],
        "from_teacher": clean["by_source"]["teacher"], "arabic": clean["by_lang"]["ar"], "english": clean["by_lang"]["en"],
        "abandoned": traces["dropped"]["abandoned: no label"], "corrections_rejected": traces["dropped"]["correction rejected by the interface"],
        "teacher_yield": f"{teacher['yield']:.1%}", "audit_n": 30, "audit_wrong": teacher["audit_wrong_of_30"], "audit_bound": "11%",
        "personal_data": ", ".join(f"{v} {KINDS.get(k, k)}" for k, v in clean["personal_data_masked" if "personal_data_masked" in clean else "personal_data_found"].items()),
        **_three()}
    print("dataset_facts is not written yet: using the reference facts")
dataset_card = fill("dataset_card.md", d_facts)
(out / "dataset_card.md").write_text(dataset_card, encoding="utf-8")
print(dataset_card)

# %% [markdown]
# ## 2 · Fill the model card from the evaluation (6')
#
# A model card answers: what is this, what is it for and not for, how was it trained, how well does it
# do **by slice**, what got worse, and under which licence may it be used.
#
# ✏️ **Exercise** · Complete `interval_text`: from one comparison of Lab 4.1, return the text
# `"+12.3 points [+9.0, +15.5]"` (the difference and its interval, in percentage points, one decimal).

# %%
def interval_text(p):
    # BEGIN SOLUTION
    return f"{p['difference'] * 100:+.1f} points [{p['low'] * 100:+.1f}, {p['high'] * 100:+.1f}]"
    # END SOLUTION

# %%
try:
    assert interval_text({"difference": 0.123, "low": 0.09, "high": 0.155}) == "+12.3 points [+9.0, +15.5]"
except NotImplementedError:
    def interval_text(p):
        return f"{p['difference'] * 100:+.1f} points [{p['low'] * 100:+.1f}, {p['high'] * 100:+.1f}]"
    print("interval_text is not written yet: using the reference version")
find = lambda scope, b: next(c for c in comparison["comparisons"] if c["scope"] == scope and c["b"] == b)
scores, of = comparison["scores"], comparison["of"]
m_facts = {
    "base_name": "the course model, models/qayd-tiny (about 1 million parameters, the course's own; not for use outside the course)",
    "precision": "4-bit base, rank-8 adapter on the query and value projections", "steps": run["steps"], "lr": run["lr"], "batch": run["batch"], "seed": run["seed"],
    "train_rows": run["train_rows"], "trainable": f"{run['trainable']:,}", "seconds": run["seconds"],
    "tuned": scores["tuned"], "base": scores["base"], "large": scores["large"], "of": of,
    "vs_base": interval_text(find("all", "base")), "vs_large": interval_text(find("all", "large")),
    "ar_tuned": comparison["by_lang"]["ar"]["tuned"], "ar_n": comparison["by_lang"]["ar"]["n"], "ar_vs_large": interval_text(find("Arabic", "large")),
    "en_tuned": comparison["by_lang"]["en"]["tuned"], "en_n": comparison["by_lang"]["en"]["n"], "en_vs_large": interval_text(find("English", "large")),
    "general_base": checks["base"]["general"]["right"], "general_tuned": checks["tuned"]["general"]["right"], "general_of": checks["base"]["general"]["of"],
    "refusal_base": checks["base"]["refusal"]["refused"], "refusal_tuned": checks["tuned"]["refusal"]["refused"], "refusal_of": checks["base"]["refusal"]["of"],
    "general_replay": checks["replay"]["general"]["right"], "refusal_replay": checks["replay"]["refusal"]["refused"], "test_replay": checks["replay"]["test"]["right"],
}
model_card = fill("model_card.md", m_facts)
(out / "model_card.md").write_text(model_card, encoding="utf-8")
print(model_card)

# %% [markdown]
# ## 3 · Write the decision memo (8')
#
# One page, for the person who decides. It states the decision first, then the four things the decision
# weighs: quality, cost, licence and data placement. Every number carries its source.
#
# ✏️ **Exercise** · Complete `decide`: return `(decision, conditions)`. `decision` is `"ship"`,
# `"ship with conditions"` or `"do not ship"`. Write the rule you would defend. The reference rule: do
# not ship if the tuned model is shown to be worse than the larger prompted model **and** the data may
# leave the chamber anyway; otherwise ship, with a condition for every regression that a check found.

# %%
def decide(evidence):
    conditions = []
    # BEGIN SOLUTION
    if evidence["worse_than_large"] and not evidence["data_must_stay"]:
        return "do not ship", ["a hosted larger model is better and allowed: use it"]
    if evidence["refusals_lost"]:
        still = evidence["replay_still_answered"]
        gate = f"only once it refuses every request of the check (it still answers {still})" if still else "which refuses every request of the check"
        conditions.append(f"{evidence['refusals_lost']} requests for personal data were answered with a tool call: do not release this adapter; release the one trained with replay, {gate}, and keep the refusal check in the release gate")
    if evidence["general_after"] < 0.95 * evidence["general_before"]:
        conditions.append(f"the general check fell from {evidence['general_before']} to {evidence['general_after']}: state it in the model card and restrict the endpoint to registration requests")
    if evidence["worse_than_large"]:
        conditions.append("the larger prompted model is better on the test set: the premium in quality is the price of keeping the data on the chamber's servers, and the memo says so")
    return ("ship with conditions" if conditions else "ship"), conditions
    # END SOLUTION

# %%
vs_large = find("all", "large")
evidence = {"worse_than_large": vs_large["high"] < 0, "better_than_base": find("all", "base")["low"] > 0, "data_must_stay": True,
            "refusals_lost": checks["base"]["refusal"]["refused"] - checks["tuned"]["refusal"]["refused"],
            "replay_still_answered": checks["replay"]["refusal"]["of"] - checks["replay"]["refusal"]["refused"],
            "general_before": checks["base"]["general"]["right"], "general_after": checks["tuned"]["general"]["right"]}
try:
    decision, conditions = decide(evidence)
except NotImplementedError:
    decision, conditions = "ship with conditions", ["decide is not written yet: write your rule, then read the reference memo in solutions/"]
    print("decide is not written yet")
memo_facts = {
    **m_facts, "decision": decision.upper(), "conditions": "\n".join(f"- {c}" for c in conditions) or "- none",
    "hosted_cost": f"{hosted:.2f}", "owned_cost": f"{load['sar_per_1000_expected']:,.2f}", "monthly": f"{load['monthly_sar']:,.0f}",
    "capacity_cost": f"{load['sar_per_1000_capacity']:.4f}", "measured_here": "measured on this machine" if load.get("measured_here") else "the course's own measurement",
    "licence_line": "Course model: the Academy's own. For a real base, the licence table of Lab 1.3 decides: Apache-2.0 allows the chamber's three uses; a revenue-capped or non-commercial licence does not.",
    "placement_line": "Members' requests are classified Restricted and stay on the chamber's servers: the hosted larger model is not an option for production, only a reference."}
memo = fill("decision_memo.md", memo_facts)
(out / "decision_memo.md").write_text(memo, encoding="utf-8")
print(memo)

# %% [markdown]
# Now the check that makes the three documents worth signing: **every number in them must be in a
# source file**. A card with a number nobody can trace is worse than a card with a gap.

# %%
def numbers(text):
    return set(re.findall(r"\d[\d,]*\.?\d*", text))


sources = " ".join(str(x) for x in (traces, teacher, clean, run, checks, comparison, load, cost, d_facts, memo_facts))
# numbers that are names or fixed parts of the templates: lab numbers, a licence version, the audit size, the dedup threshold
allowed = numbers(sources) | {"1", "2", "3", "4", "8", "30", "95", "100", "1,000", "400", "11", "0.9", "1.3", "2.0"}
for name, text in (("dataset card", dataset_card), ("model card", model_card), ("decision memo", memo)):
    loose = sorted(n for n in numbers(text) if n.rstrip(".").replace(",", "") not in {a.rstrip(".").replace(",", "") for a in allowed})
    print(f"{name}: {len(numbers(text))} numbers, not traced to a source: {loose or 'none'}")

# %% [markdown]
# 👀 **Look at** any number the check lists. It is either a rounding the template did (fine: say so) or
# a number someone typed. In your capstone the same check runs on your own three documents.

# %% [markdown]
# ## 4 · Choose the capstone task and its baseline (5')
#
# Tomorrow you adapt a small model for a task of **your** choice and defend the decision in ten minutes.
# The course's kit is your starter: the data pipeline of Day 2, the training code of Day 3, the
# evaluation and the server of Day 4. A task fits one day when it has the shape of the chamber's: one
# input, one structured output a program can check, and data you are allowed to use.
#
# ✏️ **Exercise** · Fill the plan. The cell checks that nothing is left open.

# %%
plan = {
    "task": "",                # ✏️ one sentence: from what input to what output
    "output_check": "",        # ✏️ how a program decides that an output is right
    "data_source": "",         # ✏️ where 300 or more examples come from, and why you may use them
    "test_set": "",            # ✏️ how the test set is kept apart from the tuning set
    "baseline": "",            # ✏️ the prompted baseline you will freeze first, and the larger reference model
    "placement": "",           # ✏️ the classification of the data and where the model must run
    "regression_check": "",    # ✏️ one thing the model must not lose
}
open_fields = [k for k, v in plan.items() if not v.strip()]
if open_fields:
    print("still open:", ", ".join(open_fields))
    print("Three tasks that fit, if you have none: route support tickets to one of six queues with a reason code;")
    print("extract the eight fields of a purchase order as JSON; turn a meeting request into one calendar tool call.")
else:
    Q.write_json(out / "capstone_plan.json", plan)
    print("saved", (out / "capstone_plan.json").relative_to(Q.LABS))

# %% [markdown]
# ## ✱ Stretch · Challenge the weakest number
#
# Exchange memos with a neighbour. Find the one number the decision leans on most and ask: where does it
# come from, how wide is its interval, and what would change the decision? Write the answer under the
# memo.

# %% [markdown]
# ## Check yourself
#
# 1. Which sentence of the memo would you defend first if the larger model scores higher than the tuned one?
# 2. What is the difference between the dataset card's "0 personal data found" and "no personal data"?
# 3. Why does the model card report by language and not only overall?
#
# <details><summary>Answers</summary>
#
# 1. The placement sentence: the larger model is a reference, not an option, because the requests may
#    not leave the chamber. The memo states what that requirement costs in quality and in riyals.
# 2. The first is a statement about a scan with four kinds and one list of names; the second is a claim
#    nobody can support.
# 3. Because the service is bilingual and an overall score can hide a weaker language. A reader who
#    serves Arabic-speaking members needs the Arabic line.
#
# </details>
#
# **Carry forward.** `artifacts/4.4/`: three documents and your capstone plan. Bring the plan tomorrow.
