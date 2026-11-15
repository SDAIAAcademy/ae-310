# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 2.2 · Teacher and verifier
#
# Day 2 · Session 2 (Teacher generation and verifier filtering) · 25 min.
#
# The traces mostly show what Qayd already does. To teach it what it does not do, you need labels for
# requests the desk never corrected. A larger model can write them: on Day 1 it was right far more often
# than the small one. But a teacher is wrong sometimes, and whatever enters the tuning set is what the
# student will learn. You label 300 requests with the larger model, build a verifier from schema checks
# and rule checks, and measure both the yield and what the verifier lets through.
#
# **Reference state.** `data/pool_unlabelled.jsonl` (the first 300 of 900 unlabelled requests from the
# pilot), the frozen instructions in `prompts.py`, and `data/reference/audit_30.jsonl` (the registry
# desk's audit of 30 examples). Nothing from Lab 2.1.
#
# **Tasks**
# - [ ] 1 · Label 300 requests with the larger model (4')
# - [ ] 2 · Schema checks (5')
# - [ ] 3 · Two rule checks against the request (8')
# - [ ] 4 · Yield, and what was rejected (4')
# - [ ] 5 · Audit 30 kept examples (4')
# - [ ] ✱ Stretch · Add a third rule and measure whether it rejects a right answer

# %%
#@setup
Q.banner("Lab 2.2")

# %%
import math
import re
from collections import Counter

import prompts

pool = Q.load("pool_unlabelled")[:300]
REF = Q.reference("d2_2_teacher")
val = {r["id"]: r for r in Q.load("requests_val")}
shots = [{"text": val[i]["text"], "call": val[i]["gold"]} for i in prompts.EXAMPLE_IDS]
print(len(pool), "unlabelled requests")

# %% [markdown]
# ## 1 · Label 300 requests with the larger model (4')
#
# The teacher gets the **same frozen instructions and examples** as the baseline. A label is only useful
# to the student if it is in the format the student must produce.
#
# The requests in the pool are synthetic. With real members' requests this step would send Restricted
# data to a hosted model, which the chamber does not allow: a real project runs the teacher on its own
# servers, or labels synthetic and masked requests only.

# %%
teacher = Q.run(pool, prompts.FROZEN, "large", shots)
print("answered by:", Counter(r["source"] for r in teacher))
for i in (0, 1):
    print(pool[i]["text"][:100])
    print("   teacher:", teacher[i]["text"].replace("\n", " "))

# %% [markdown]
# ## 2 · Schema checks (5')
#
# The first filter asks only: would the registry accept this? `C.parse_call` returns `None` when there is
# no call that parses, and `C.schema_errors(call)` lists the interface's objections.
#
# ✏️ **Exercise** · Complete `schema_check`: return the list of schema errors of a teacher output (an
# empty list when it is a call the interface accepts).

# %%
def schema_check(output):
    # BEGIN SOLUTION
    return C.schema_errors(C.parse_call(output))
    # END SOLUTION

# %%
# Check, with a safety net
try:
    assert schema_check("I renewed it.") and not schema_check(C.render_call(C.call("renew_membership", cr_number="4012345678", years=1)))
except NotImplementedError:
    def schema_check(output):
        return C.schema_errors(C.parse_call(output))
    print("schema_check is not written yet: using the reference version")
schema_failed = [i for i, r in enumerate(teacher) if schema_check(r["text"])]
print(len(schema_failed), "of", len(pool), "teacher outputs fail the schema")
for i in schema_failed[:3]:
    print("  ", teacher[i]["text"].replace("\n", " ")[:110], "->", schema_check(teacher[i]["text"])[0])

# %% [markdown]
# ## 3 · Two rule checks against the request (8')
#
# A schema knows the interface, not the request. `{"cr_number": "4000000000"}` is a perfectly valid
# argument, and an invention. A **rule check** compares the call with the text it claims to answer.
#
# ✏️ **Exercise** · Complete the two rules in `rule_check`:
#
# 1. a `cr_number` in the call must appear in the request (after converting Arabic-Indic digits with
#    `C.western_digits`);
# 2. a call that escalates with the reason `missing_cr` is wrong when the request does contain a
#    registration number.
#
# Two more rules are written for you: the company name and the city must appear in the request.

# %%
CR = re.compile(r"(?<![0-9])40[0-9]{8}(?![0-9])")


def rule_check(text, call):
    """Why a schema-valid call disagrees with its request; an empty list when no rule objects."""
    errors = []
    numbers = CR.findall(C.western_digits(text))  # the registration numbers the request contains
    args = call["arguments"]
    # BEGIN SOLUTION
    if "cr_number" in args and args["cr_number"] not in numbers:
        errors.append("the registration number is not in the request")
    if call["name"] == "escalate_to_staff" and args.get("reason") == "missing_cr" and numbers:
        errors.append("escalated as missing although the request gives a number")
    # END SOLUTION
    if call["name"] == "create_membership" and args.get("company_name", "") not in text:
        errors.append("the company name is not in the request")
    city = args.get("city") or (args.get("value") if args.get("field") == "city" else None)
    if city and city not in text and C.CITY_AR.get(city, "\0") not in text:
        errors.append("the city is not in the request")
    return errors

# %%
def _rule_check(text, call):
    errors, numbers, args = [], CR.findall(C.western_digits(text)), call["arguments"]
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

try:
    probe = C.call("renew_membership", cr_number="4000000000", years=1)
    assert rule_check("Please renew our membership for two years.", probe), "rule 1: an invented number must be caught"
    assert not rule_check("Renew CR ٤٠١٢٣٤٥٦٧٨ for one year.", C.call("renew_membership", cr_number="4012345678", years=1)), "convert the digits first"
    assert rule_check("Renew CR 4012345678.", C.call("escalate_to_staff", reason="missing_cr")), "rule 2"
except NotImplementedError:
    rule_check = _rule_check
    print("rule_check is not written yet: using the reference version")


def verify(text, output):
    errors = schema_check(output)
    return ("schema", errors) if errors else ("rule", rule_check(text, C.parse_call(output)))

# %% [markdown]
# ## 4 · Yield, and what was rejected (4')

# %%
kept, rejected = [], Counter()
for row, r in zip(pool, teacher):
    stage, errors = verify(row["text"], r["text"])
    if errors:
        rejected[stage] += 1
        continue
    kept.append({"id": row["id"], "source": "teacher", "lang": row["lang"], "messages": [
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": C.parse_call(r["text"])}]}]})
yield_ = len(kept) / len(pool)
Q.show("The verifier on 300 teacher outputs", [("rejected by the schema", rejected["schema"]), ("rejected by a rule", rejected["rule"]),
                                              ("kept", len(kept)), ("yield", f"{yield_:.1%}")])
if Q.using_standin():
    assert len(kept) == REF["kept"] and dict(rejected) == REF["rejected"], (len(kept), dict(rejected))
out = Q.artifacts("2.2")
Q.write_jsonl(out / "examples_teacher.jsonl", kept)
print("saved", (out / "examples_teacher.jsonl").relative_to(Q.LABS))

# %% [markdown]
# ## 5 · Audit 30 kept examples (4')
#
# A high yield says the verifier rejected little. It does not say that what it kept is right. The only
# way to know is for a person to check a sample. Khuzama's desk audited 30 of the kept examples.
#
# ✏️ **Exercise** · Complete `audit_bound`: given `wrong` errors found in `n` audited examples, return
# the upper end of the 95% Wilson interval for the true error rate (the formula of Lab 1.2).

# %%
def audit_bound(wrong, n, z=1.96):
    # BEGIN SOLUTION
    p = wrong / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre + half
    # END SOLUTION

# %%
audit = Q.reference("audit_30")
wrong = sum(a["desk_says"] == "wrong" for a in audit)
try:
    bound = audit_bound(wrong, len(audit))
    assert abs(audit_bound(0, 30) - 0.1135) < 0.001, audit_bound(0, 30)
except NotImplementedError:
    bound = 0.1135 if wrong == 0 else float("nan")
    print("audit_bound is not written yet: using the reference result")
print(f"The desk found {wrong} wrong in {len(audit)}. Upper bound on the error rate of the kept examples: {bound:.1%}")
print(f"The answer key (which no real project has): {REF['wrong_but_kept']} of the {REF['kept']} kept examples are wrong, {REF['wrong_but_kept'] / REF['kept']:.1%}.")
print(f"The teacher was wrong on {REF['teacher_wrong']} of 300; the verifier caught {REF['teacher_wrong'] - REF['wrong_but_kept']}.")

# %% [markdown]
# 👀 **Look at** the three lines together. The audit found nothing, and "nothing in 30" allows an error
# rate of up to about 11%. The truth here is under 3%: errors that are valid calls, with a number that is
# in the request, and wrong (the partner's number, the wrong tool, the wrong number of years). No schema
# and no rule of this kind can see them. Two conclusions for the card: state the audit size with its
# bound, and expect the student to inherit these errors (Lab 3.4 goes looking for one).

# %% [markdown]
# ## ✱ Stretch · A third rule
#
# Add a rule and measure what it costs. A strict rule can reject right answers, and each right answer
# rejected is an example lost. One to try: "a request that mentions two registration numbers is
# rejected", which removes every case where the teacher could have taken the wrong one.

# %%
two_numbers = [e for e in kept if len(set(CR.findall(C.western_digits(e["messages"][0]["content"])))) >= 2]
print(len(two_numbers), "kept examples mention two registration numbers. The rule would remove all of them,")
print("right or wrong, and with them every example that teaches the student which number to take.")

# %% [markdown]
# ## Check yourself
#
# 1. Why must the teacher use the frozen instructions and not a better prompt written for it?
# 2. The yield is 98%. What does that tell you about the quality of the kept examples?
# 3. An audit of 30 finds no error. What do you write in the dataset card?
#
# <details><summary>Answers</summary>
#
# 1. The label has to be the exact format the student must produce. A better prompt is fine as long as
#    the output format is identical; a different format teaches the student the wrong thing.
# 2. Nothing. Yield measures how much the verifier rejected, not whether what passed is right.
# 3. "0 errors in an audit of 30; 95% upper bound 11%." And the kinds of error the verifier cannot see.
#
# </details>
#
# **Carry forward.** `artifacts/2.2/examples_teacher.jsonl`. The supplied `data/tuning/teacher_filtered.jsonl`
# holds the same pipeline run over all 900 pool requests; Lab 3.4 distils from it.
