# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 1.3 · Three licences
#
# Day 1 · Session 3 (Licences and terms of open-weight models) · 25 min.
#
# "Open weights" says that you can download a model, not what you may do with it. Before a single
# training step, Zakariya Al-Mousa, the chamber's legal counsel, wants one line per candidate model: may
# we adapt it, may we deploy the adapted model in the registration service, may we give the adapter to
# our sister chambers, and which clause decides. You read three licence sheets and write those lines.
#
# **Reference state.** `data/licences/` holds one sheet per model: the licence, its address, the day it
# was read and the clauses that matter, in the course's own words. `data/chamber_facts.json` holds the
# facts about the chamber (fictional). A sheet is a reading aid, not the licence: the address is there
# so that you open the text itself.
#
# **Tasks**
# - [ ] 1 · Read three licence sheets (6')
# - [ ] 2 · Record adapt, deploy and distribute for each (8')
# - [ ] 3 · Check the chamber's facts against each condition (6')
# - [ ] 4 · Write the licence line of the memo (5')
# - [ ] ✱ Stretch · Fill a sheet for a fourth model of your choice from its licence text on the Hub

# %%
#@setup
Q.banner("Lab 1.3")

# %%
sheets = {p.stem: Q.read_json(p) for p in sorted((Q.DATA / "licences").glob("*.json"))}
facts = Q.read_json(Q.DATA / "chamber_facts.json")
print("candidates:", ", ".join(s["model"] for s in sheets.values()))

# %% [markdown]
# ## 1 · Read three licence sheets (6')
#
# 👀 **Look at** each sheet and open its `licence_url`. For each model, find: the grant (what you may
# do), the conditions on passing the model on, and any clause that depends on **who** uses it or **for
# what**.

# %%
for key, s in sheets.items():
    print("=" * 100)
    print(f"{s['model']}  ·  {s['parameters']} parameters  ·  {s['slot']}")
    print(f"licence: {s['licence']}  (read on {s['read_on']})")
    print(f"         {s['licence_url']}")
    print(f"access : {s['access']}")
    print(f"Arabic : {s['arabic']}")
    for c in s["clauses"]:
        print(f"   section {c['id']:>4}: {c['says']}")

# %% [markdown]
# ## 2 · Record adapt, deploy and distribute for each (8')
#
# Three separate permissions:
#
# - **adapt**: fine-tune the model, which makes a derivative work;
# - **deploy**: run the adapted model in the registration service, which members pay fees for;
# - **distribute**: give the trained adapter, a derivative work, to two sister chambers.
#
# Answer each with `"yes"`, `"no"` or `"ask"` (the text does not settle it for this chamber: counsel, or
# the licensor, must). Name the section that decides.
#
# ✏️ **Exercise** · Complete `record`. The chamber's plan is in `facts["plan"]`.

# %%
print(facts["legal_form"], facts["revenue_sources"])
for k, v in facts["plan"].items():
    print(f"  {k}: {v}")


def record():
    """One entry per sheet key: {"adapt", "deploy", "distribute", "section"}."""
    # BEGIN SOLUTION
    return {
        "qwen3-0.6b": {"adapt": "yes", "deploy": "yes", "distribute": "yes", "section": "2 and 4: derivative works and redistribution are granted, with the licence and a notice of changes"},
        "lfm3-1.7b": {"adapt": "yes", "deploy": "ask", "distribute": "ask", "section": "5: commercial use is licensed only below the revenue threshold"},
        "command-r7b-arabic": {"adapt": "yes", "deploy": "ask", "distribute": "ask", "section": "2: every right is for NonCommercial purposes only; and the gated page was not read"},
    }
    # END SOLUTION

# %%
# Check, with a safety net
ACCEPTED = {
    "qwen3-0.6b": {"adapt": {"yes"}, "deploy": {"yes"}, "distribute": {"yes"}},
    "lfm3-1.7b": {"adapt": {"yes"}, "deploy": {"ask", "no"}, "distribute": {"ask", "no"}},
    "command-r7b-arabic": {"adapt": {"yes", "ask"}, "deploy": {"ask", "no"}, "distribute": {"ask", "no"}},
}
try:
    verdicts = record()
    for key, allowed in ACCEPTED.items():
        for what, values in allowed.items():
            assert verdicts[key][what] in values, f"{key}: read the sheet again for '{what}' (you wrote {verdicts[key][what]!r})"
        assert verdicts[key]["section"].strip(), f"{key}: name the section that decides"
except NotImplementedError:
    verdicts = {
        "qwen3-0.6b": {"adapt": "yes", "deploy": "yes", "distribute": "yes", "section": "2 and 4"},
        "lfm3-1.7b": {"adapt": "yes", "deploy": "ask", "distribute": "ask", "section": "5"},
        "command-r7b-arabic": {"adapt": "yes", "deploy": "ask", "distribute": "ask", "section": "2"},
    }
    print("record is not written yet: showing the supplied table")
Q.show("Licence table", [("model", "adapt", "deploy", "distribute", "deciding section")] +
       [(sheets[k]["model"], v["adapt"], v["deploy"], v["distribute"], v["section"]) for k, v in verdicts.items()])

# %% [markdown]
# ## 3 · Check the chamber's facts against each condition (6')
#
# Two conditions depend on facts about the chamber, and a licence table without the facts is an opinion.
#
# ✏️ **Exercise** · Complete `above_threshold`: is the chamber's annual revenue, converted to United
# States dollars, at or above the threshold of the LFM Open License?

# %%
def above_threshold(revenue_sar, sar_per_usd, threshold_usd=10_000_000):
    # BEGIN SOLUTION
    return revenue_sar / sar_per_usd >= threshold_usd
    # END SOLUTION

# %%
try:
    over = above_threshold(facts["annual_revenue_sar"], facts["sar_per_usd"])
    assert above_threshold(37_500_000, 3.75) is True and above_threshold(37_499_999, 3.75) is False
except NotImplementedError:
    over = facts["annual_revenue_sar"] / facts["sar_per_usd"] >= 10_000_000
    print("above_threshold is not written yet: using the reference result")
usd = facts["annual_revenue_sar"] / facts["sar_per_usd"]
print(f"annual revenue: SAR {facts['annual_revenue_sar']:,} = USD {usd:,.0f}  ->  at or above the threshold: {over}")
assert over is True
print()
print("Fact 1 · The chamber is above the threshold. Whether its use is licensed now turns on one question:")
print("         is running the model inside a service that members pay fees for a 'Commercial Use'?")
print("Fact 2 · The chamber is a non-profit body. Under CC BY-NC the same question returns in other words:")
print("         is the use 'primarily intended for or directed towards commercial advantage or monetary compensation'?")
print("Neither text answers for a fee-charging non-profit. That is what 'ask' means: a question for counsel")
print("or for the licensor, written down before training starts, not discovered after deployment.")

# %% [markdown]
# ## 4 · Write the licence line of the memo (5')
#
# The decision memo of Day 4 has one line on licences. Write it now, while the clauses are in front of you:
# the model you would take forward, the section that allows the chamber's plan, and what the other two
# would need first.
#
# ✏️ **Exercise** · Complete `memo_line`. It must name the chosen model and a section number.

# %%
def memo_line():
    # BEGIN SOLUTION
    return ("Take Qwen/Qwen3-0.6B forward: Apache License 2.0, sections 2 and 4, allows adapting, deploying and passing on the adapter, "
            "with the licence text and a notice of changes. LFM3-1.7B needs counsel's answer on section 5 (the chamber is above the revenue threshold). "
            "Command R7B Arabic needs a commercial agreement or counsel's answer on NonCommercial use, and its gated conditions read.")
    # END SOLUTION

# %%
try:
    line = memo_line()
    assert "Qwen" in line and any(ch.isdigit() for ch in line.replace("Qwen3-0.6B", "").replace("LFM3-1.7B", "").replace("R7B", "")), "name the model and the section"
except NotImplementedError:
    line = "Take Qwen/Qwen3-0.6B forward (Apache License 2.0, sections 2 and 4). The other two need counsel first."
    print("memo_line is not written yet: showing the supplied line")
print(line)
out = Q.artifacts("1.3") / "licence_table.json"
Q.write_json(out, {"read_on": max(s["read_on"] for s in sheets.values()), "verdicts": verdicts, "memo_line": line})
print("saved", out.relative_to(Q.LABS))

# %% [markdown]
# ## ✱ Stretch · A fourth model
#
# Choose a small model on the Hub that you might use in the capstone. Open its licence file, not only the
# tag on its card, and write a sheet with the same keys as the three above. Two things to look for that
# these three do not show: an **acceptable-use policy** that is a separate document, and conditions on the
# **outputs** of the model (which matter when the model is your teacher on Day 3).

# %% [markdown]
# ## Check yourself
#
# 1. A model card carries the tag "apache-2.0". What have you verified?
# 2. Which of the three permissions does a licence most often restrict first?
# 3. The sheet for the third model says its page was not read. Why does that matter?
#
# <details><summary>Answers</summary>
#
# 1. That someone typed a tag. The licence file in the repository is the text; read it, and record the
#    day you read it, because a later release can change it.
# 2. Deployment for commercial purposes, then distribution. Adapting for research is the most widely
#    allowed.
# 3. A gated model can add conditions on its page that are not in the licence text. A verdict that
#    leaves them out is incomplete, and "ask" is the honest entry.
#
# </details>
#
# **Carry forward.** `artifacts/1.3/licence_table.json`: the licence line of the decision memo (Lab 4.4).
