# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 1.4 · Cost of two options
#
# Day 1 · Session 4 (Cost and placement of the options) · 25 min.
#
# Maysoon Al-Dabbagh, who will sign the decision, asks two questions: what does each option cost per
# thousand requests, and where is a member's request processed? You estimate both for a large hosted
# model and for a tuned small model on the chamber's own accelerator, find the volume at which they cost
# the same, and write down what the numbers do and do not decide.
#
# **Reference state.** `data/prices.json`: published prices of three hosted models, with the day they
# were read. `data/onprem.json`: the chamber's planning figures (fictional assumptions of the case
# study). `data/requests_100.jsonl`. Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · Measure tokens per request (5')
# - [ ] 2 · Cost per thousand requests on a hosted model (5')
# - [ ] 3 · Cost per thousand on an owned accelerator (7')
# - [ ] 4 · Break-even volume and placement (8')
# - [ ] ✱ Stretch · Add the one-off cost of tuning and find the month in which it is paid back

# %%
#@setup
Q.banner("Lab 1.4")

# %%
import statistics

import prompts

requests = Q.load("requests_100")
prices = Q.read_json(Q.DATA / "prices.json")
onprem = Q.read_json(Q.DATA / "onprem.json")
SAR_PER_USD = 3.75
print(prices["note"])
print("read on", prices["read_on"], "from", prices["source"])
for m in prices["models"]:
    print(f"  {m['id']:26s} input {m['input_per_million']:.3f}  output {m['output_per_million']:.3f}  USD per million tokens  ({m['role']})")
print()
print(onprem["note"])

# %% [markdown]
# ## 1 · Measure tokens per request (5')
#
# A hosted model is paid by the token, and a prompted model reads its whole instructions on **every**
# request: the schema hints, the code table, the rules and the four examples of the frozen baseline. A
# tuned model needs none of that text; what it learnt is in the adapter.
#
# `Q.tokens(text)` is a rough count (about four characters of English or two and a half of Arabic for
# one token). A provider's own count on a real call replaces it.
#
# ✏️ **Exercise** · Complete `tokens_per_request`: the mean input tokens of a prompted request (the
# instructions, the examples and the request), the mean input tokens of a tuned model's request (the
# request alone), and the mean output tokens (the correct call).

# %%
val = {r["id"]: r for r in Q.load("requests_val")}
examples_text = "".join(val[i]["text"] + C.render_call(val[i]["gold"]) for i in prompts.EXAMPLE_IDS)


def tokens_per_request(rows):
    """Return (prompted_input, tuned_input, output): mean tokens per request."""
    fixed = Q.tokens(prompts.FROZEN) + Q.tokens(examples_text)
    # BEGIN SOLUTION
    request = statistics.mean(Q.tokens(r["text"]) for r in rows)
    output = statistics.mean(Q.tokens(C.render_call(r["gold"])) for r in rows)
    return fixed + request, request, output
    # END SOLUTION

# %%
# Check, with a safety net
try:
    prompted_in, tuned_in, out_tokens = tokens_per_request(requests)
    assert prompted_in > 10 * tuned_in > 0 and 15 < out_tokens < 80, (prompted_in, tuned_in, out_tokens)
except NotImplementedError:
    tuned_in = statistics.mean(Q.tokens(r["text"]) for r in requests)
    out_tokens = statistics.mean(Q.tokens(C.render_call(r["gold"])) for r in requests)
    prompted_in = Q.tokens(prompts.FROZEN) + Q.tokens(examples_text) + tuned_in
    print("tokens_per_request is not written yet: using the reference result")
print(f"prompted request: {prompted_in:6.0f} input tokens   tuned request: {tuned_in:4.0f} input tokens   answer: {out_tokens:3.0f} output tokens")
print(f"The instructions are {prompted_in / tuned_in:.0f} times the request.")

# %% [markdown]
# ## 2 · Cost per thousand requests on a hosted model (5')
#
# ✏️ **Exercise** · Complete `hosted_per_thousand`: the cost in riyals of 1,000 requests at a model's
# published prices (United States dollars per million tokens; SAR 3.75 to the dollar).

# %%
def hosted_per_thousand(model, input_tokens, output_tokens):
    # BEGIN SOLUTION
    usd = 1000 * (input_tokens * model["input_per_million"] + output_tokens * model["output_per_million"]) / 1_000_000
    return usd * SAR_PER_USD
    # END SOLUTION

# %%
try:
    assert abs(hosted_per_thousand({"input_per_million": 1.0, "output_per_million": 2.0}, 1000, 500) - 7.5) < 1e-9
    hosted = {m["id"]: hosted_per_thousand(m, prompted_in, out_tokens) for m in prices["models"]}
except NotImplementedError:
    hosted = {m["id"]: 1000 * (prompted_in * m["input_per_million"] + out_tokens * m["output_per_million"]) / 1e6 * SAR_PER_USD for m in prices["models"]}
    print("hosted_per_thousand is not written yet: using the reference result")
for name, sar in hosted.items():
    print(f"  {name:26s} SAR {sar:6.2f} per 1,000 requests")
LARGE = prices["models"][0]["id"]

# %% [markdown]
# ## 3 · Cost per thousand on an owned accelerator (7')
#
# An owned accelerator costs the same whether it answers one request or a million: the purchase spread
# over its useful life, the electricity, and the hours of the engineer who keeps it running. The cost per
# request is that monthly sum divided by the volume.
#
# ✏️ **Exercise** · Complete `onprem_monthly`: the monthly cost in riyals from the planning figures.
# Count 730 hours in a month.

# %%
for k, v in onprem.items():
    if k not in ("note", "tuning_one_off"):
        print(f"  {k}: {v}")


def onprem_monthly(f):
    # BEGIN SOLUTION
    hardware = (f["accelerator_purchase_sar"] + f["server_and_installation_sar"]) / (f["useful_life_years"] * 12)
    power = f["power_kw"] * 730 * f["electricity_sar_per_kwh"]
    people = f["operations_hours_per_month"] * f["engineer_sar_per_hour"]
    return hardware + power + people
    # END SOLUTION

# %%
try:
    monthly = onprem_monthly(onprem)
    assert abs(onprem_monthly({"accelerator_purchase_sar": 48000, "server_and_installation_sar": 0, "useful_life_years": 4, "power_kw": 1,
                               "electricity_sar_per_kwh": 0.1, "operations_hours_per_month": 1, "engineer_sar_per_hour": 100}) - 1173) < 1e-6
except NotImplementedError:
    monthly = ((onprem["accelerator_purchase_sar"] + onprem["server_and_installation_sar"]) / (onprem["useful_life_years"] * 12)
               + onprem["power_kw"] * 730 * onprem["electricity_sar_per_kwh"] + onprem["operations_hours_per_month"] * onprem["engineer_sar_per_hour"])
    print("onprem_monthly is not written yet: using the reference result")
volume = onprem["expected_requests_per_month"]
capacity = onprem["requests_per_second_at_capacity"] * 730 * 3600
own_per_thousand = monthly / (volume / 1000)
print(f"owned accelerator: SAR {monthly:,.0f} a month, whatever the volume")
print(f"at {volume:,} requests a month: SAR {own_per_thousand:,.2f} per 1,000 requests")
print(f"capacity at {onprem['requests_per_second_at_capacity']} requests a second: {capacity:,.0f} a month; the chamber would use {volume / capacity:.2%} of it")

# %% [markdown]
# ## 4 · Break-even volume and placement (8')
#
# ✏️ **Exercise** · Complete `break_even`: the monthly volume at which the owned accelerator costs the
# same per request as the hosted model.

# %%
def break_even(monthly_sar, hosted_sar_per_thousand):
    """Requests a month at which both options cost the same."""
    # BEGIN SOLUTION
    return monthly_sar / (hosted_sar_per_thousand / 1000)
    # END SOLUTION

# %%
try:
    even = break_even(monthly, hosted[LARGE])
    assert abs(break_even(1000, 10) - 100_000) < 1e-6
except NotImplementedError:
    even = monthly / (hosted[LARGE] / 1000)
    print("break_even is not written yet: using the reference result")
Q.show("Two options at the chamber's volume", [
    ("option", "SAR per 1,000 requests", "where a request is processed", "meets the placement requirement"),
    (f"large hosted model ({LARGE})", f"{hosted[LARGE]:.2f}", "the provider's servers, outside the chamber", "no"),
    ("tuned small model, own accelerator", f"{own_per_thousand:.2f}", "the chamber's own servers", "yes"),
])
print(f"\nbreak-even: {even:,.0f} requests a month. The chamber expects {volume:,}: {even / volume:,.0f} times less.")
out = Q.artifacts("1.4") / "cost_table.json"
Q.write_json(out, {"prices_read_on": prices["read_on"], "hosted_sar_per_thousand": hosted, "onprem_monthly_sar": monthly,
                   "onprem_sar_per_thousand": own_per_thousand, "break_even_requests_per_month": even, "volume": volume})
print("saved", out.relative_to(Q.LABS))

# %% [markdown]
# 👀 **Look at** the two numbers side by side, and do not soften them. At the chamber's volume the hosted
# model is far cheaper per request, and the chamber would have to grow by orders of magnitude before
# owning hardware paid for itself. **Cost does not justify the small model here.** What decides is the
# last column: the requests carry members' personal data, the chamber classifies them as Restricted, and
# its requirement is that they stay on its own servers. The memo must say exactly that: the chamber pays
# a premium for placement, this is the premium, and this is why it is worth paying. A memo that claims
# the small model is cheaper would be wrong, and Maysoon would find out.

# %% [markdown]
# ## ✱ Stretch · The one-off cost of tuning

# %%
t = onprem["tuning_one_off"]
one_off = t["engineer_hours"] * onprem["engineer_sar_per_hour"] + t["gpu_hours"] * t["gpu_sar_per_hour"]
print(f"tuning, once: SAR {one_off:,.0f} ({t['engineer_hours']} engineer hours and {t['gpu_hours']} GPU hours)")
print(f"that is {one_off / monthly:.1f} months of running the accelerator; almost all of it is the engineer's time")
print("Against a prompted small model on the same accelerator, tuning is paid back by quality, not by money: both cost the same to serve.")

# %% [markdown]
# ## Check yourself
#
# 1. Why does a tuned model read fewer input tokens than a prompted one?
# 2. The chamber's volume doubles. What happens to the cost per thousand requests of each option?
# 3. Which line of the cost table would change if the requests were public data?
#
# <details><summary>Answers</summary>
#
# 1. The prompted model reads the schema, the code table, the rules and the examples on every request.
#    The tuned model learnt the format and the behaviour; only the request is sent. (A code table that
#    changes still belongs in the context.)
# 2. Hosted: unchanged, it is a price per token. Owned: halved, because the monthly cost is fixed.
# 3. The last column. With public data the placement requirement falls away and cost decides, and the
#    hosted model wins by a wide margin at this volume.
#
# </details>
#
# **Carry forward.** `artifacts/1.4/cost_table.json`: the cost section of the decision memo. Lab 4.3
# replaces the planning figure for requests per second with a measured one.
