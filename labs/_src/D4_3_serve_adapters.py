# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6", "transformers>=4.56", "peft>=0.17", "jinja2>=3.1", "httpx>=0.28", "fastapi>=0.115", "uvicorn>=0.34", "locust>=2.37"]
# ///
# %% [markdown]
# # Lab 4.3 · Two adapters, one base, under load
#
# Day 4 · Session 3 (One base, several adapters) · 25 min.
#
# Jarrah Al-Suhaim has one accelerator, and the chamber already wants a second assistant. One base in
# memory with a small adapter per task is how one machine serves both. You start a server that holds the
# 4-bit base and two adapters, call each by name, load-test it at three levels of concurrency, and turn
# the throughput you measure into a cost per thousand requests.
#
# **This lab uses the course model on every machine**, on a processor. `serve_adapters.py` is a small
# teaching server (about a hundred lines, worth reading): it picks the adapter from the `model` field of
# the request and answers requests that arrive together in one batch. The numbers you measure are those
# of a 1-million-parameter model on your laptop. The **method** is what you take to a real server; the
# guide gives the same steps for vLLM on an accelerator.
#
# **Reference state.** `adapters/registry-4bit/` and `adapters/registry-replay/` (supplied),
# `data/onprem.json` (Jarrah's planning figures) and `data/reference/d4_3_load.json` (the course's own
# load test, used when the server cannot start here).
#
# **Tasks**
# - [ ] 1 · Start the server and call each adapter (5')
# - [ ] 2 · Load-test at three levels of concurrency (9')
# - [ ] 3 · Read throughput and latency percentiles (5')
# - [ ] 4 · Cost per thousand requests at the target rate (6')
# - [ ] ✱ Stretch · Send the same load to the base with no adapter: what does an adapter cost in latency?

# %%
#@setup
Q.banner("Lab 4.3")

# %%
import csv
import subprocess
import time

import httpx

PORT = 8310
URL = f"http://127.0.0.1:{PORT}"
REF = Q.reference("d4_3_load")
out = Q.artifacts("4.3")
onprem = Q.read_json(Q.DATA / "onprem.json")

# %% [markdown]
# ## 1 · Start the server and call each adapter (5')
#
# The server runs as its own process, as it would in production. The cell starts it and waits until it
# answers.

# %%
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "serve_adapters:app", "--port", str(PORT), "--log-level", "warning"], cwd=Q.LABS)
LIVE = False
for _ in range(90):
    try:
        models = httpx.get(URL + "/v1/models", timeout=2).json()["data"]
        LIVE = True
        break
    except Exception:
        if server.poll() is not None:
            break
        time.sleep(1)
if LIVE:
    for m in models:
        print(m)
else:
    print("The server did not start here (a blocked port, or a notebook service without local processes).")
    print("The lab continues with the course's own measurements.")

# %% [markdown]
# 👀 **Look at** the list. One base, loaded once; each adapter is a few thousand parameters beside it.
#
# ✏️ **Exercise** · Complete `ask`: send one request to the server for the adapter `model` and return
# the JSON of the answer. The body is that of a chat completion: `{"model": ..., "messages": [{"role":
# "user", "content": ...}]}`, posted to `/v1/chat/completions`.

# %%
def ask(model, text):
    # BEGIN SOLUTION
    body = {"model": model, "messages": [{"role": "user", "content": text}]}
    return httpx.post(URL + "/v1/chat/completions", json=body, timeout=60).json()
    # END SOLUTION

# %%
# Check, with a safety net
def _ask(model, text):
    return httpx.post(URL + "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": text}]}, timeout=60).json()

request = Q.load("requests_val")[1]
refusals = Q.load("refusal_check")
tuned_answers = Q.reference("d4_checks")["tuned"]["outputs"]["refusal"]  # Lab 4.2's supplied outputs
# a request for personal data that the tuned adapter answered with a tool call in Lab 4.2
personal = next(r for r, o in zip(refusals, tuned_answers) if r["lang"] == "en" and "<tool_call>" in o)
if LIVE:
    try:
        ask("qayd-registry", request["text"])
    except NotImplementedError:
        ask = _ask
        print("ask is not written yet: using the reference version")
    for text in (request["text"], personal["text"]):
        print("request:", text[:110])
        for name in ("qayd-base", "qayd-registry", "qayd-replay"):
            a = ask(name, text)
            print(f"  {name:<14} {a['choices'][0]['message']['content'].replace(chr(10), ' ')[:100]}   ({a['seconds'] * 1000:.0f} ms)")

# %% [markdown]
# 👀 **Look at** the second request. Same base, same request, three behaviours, chosen by one word in
# the request body. Which adapter a caller may name is an access decision, not a technical one.

# %% [markdown]
# ## 2 · Load-test at three levels of concurrency (9')
#
# `locustfile.py` describes one simulated member: send a request to one of the two adapters, wait for
# the answer, send the next. `-u 8` runs eight of them at once, so eight requests are in flight. Each
# level runs for a fixed time and writes its statistics to a CSV file.

# %%
LEVELS = [1, 8, 32]
SECONDS = Q.pick(30, 6)


def load_test(users, models="qayd-registry,qayd-replay", label=None):
    """Run Locust without its web page and return the totals row of its statistics file."""
    prefix = out / (label or f"u{users}")
    env = {**os.environ, "LOAD_MODELS": models}
    httpx.post(URL + "/stats/reset")
    subprocess.run([sys.executable, "-m", "locust", "-f", "locustfile.py", "--headless", "-u", str(users), "-r", str(users), "-t", f"{SECONDS}s",
                    "--host", URL, "--csv", str(prefix), "--only-summary", "--loglevel", "ERROR"], cwd=Q.LABS, env=env, capture_output=True, text=True)
    with open(f"{prefix}_stats.csv", newline="", encoding="utf-8") as f:
        row = [r for r in csv.DictReader(f) if r["Name"] == "Aggregated"][0]
    batch = httpx.get(URL + "/stats").json()
    return {"users": users, "requests": int(row["Request Count"]), "failures": int(row["Failure Count"]), "per_second": round(float(row["Requests/s"]), 2),
            "p50_ms": float(row["50%"]), "p95_ms": float(row["95%"]), "p99_ms": float(row["99%"]), "mean_batch": batch["mean_batch"]}


results = []
if LIVE:
    try:
        for users in LEVELS:
            results.append(load_test(users))
            print(results[-1])
    except Exception as error:  # Locust could not run here
        print("The load test could not run here:", type(error).__name__, error)
        results = []
if not results:
    results = REF["levels"]
    print("using the course's own measurements:")
    for r in results:
        print(r)

# %% [markdown]
# ## 3 · Read throughput and latency percentiles (5')
#
# ✏️ **Exercise** · Complete `capacity`: return the result with the highest throughput among those whose
# 95th-percentile latency is at most `limit_ms` and that had no failed request. That is the server's
# capacity **under a latency target**; throughput with no target is not a capacity. Return `None` when
# no level qualifies.

# %%
def capacity(results, limit_ms):
    # BEGIN SOLUTION
    ok = [r for r in results if r["p95_ms"] <= limit_ms and r["failures"] == 0]
    return max(ok, key=lambda r: r["per_second"]) if ok else None
    # END SOLUTION

# %%
def _capacity(results, limit_ms):
    ok = [r for r in results if r["p95_ms"] <= limit_ms and r["failures"] == 0]
    return max(ok, key=lambda r: r["per_second"]) if ok else None

LIMIT_MS = 5000  # the owner's target: 95% of answers within five seconds
try:
    best = capacity(results, LIMIT_MS)
    assert best == _capacity(results, LIMIT_MS)
except NotImplementedError:
    best = _capacity(results, LIMIT_MS)
    print("capacity is not written yet: using the reference version")
print(f"{'in flight':>10}{'requests/s':>12}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'mean batch':>12}{'failures':>10}")
for r in results:
    print(f"{r['users']:>10}{r['per_second']:>12}{r['p50_ms']:>9.0f}{r['p95_ms']:>9.0f}{r['p99_ms']:>9.0f}{r['mean_batch']:>12}{r['failures']:>10}")
print(f"capacity with 95% of answers within {LIMIT_MS} ms:", f"{best['per_second']} requests/s at {best['users']} in flight" if best else "no level meets the target")
tighter = _capacity(results, LIMIT_MS - 1000)
print(f"with a target of {LIMIT_MS - 1000} ms instead:", f"{tighter['per_second']} requests/s at {tighter['users']} in flight" if tighter else "no level meets the target")

# %% [markdown]
# 👀 **Look at** the two columns together. With more requests in flight the server fills larger batches:
# throughput rises, and so does the time each member waits. A mean hides the slow answers; a member who
# waits is a 95th or 99th percentile. The latency target is a decision of the service owner, and it sets
# the capacity.

# %% [markdown]
# ## 4 · Cost per thousand requests at the target rate (6')
#
# Jarrah's monthly cost of running the machine does not depend on how many requests it answers. So the
# cost per thousand requests depends on how busy the machine is.
#
# ✏️ **Exercise** · Complete `cost_per_thousand`: the monthly cost divided by the thousands of requests
# answered in the month.

# %%
def cost_per_thousand(monthly_sar, requests_per_month):
    # BEGIN SOLUTION
    return monthly_sar / (requests_per_month / 1000)
    # END SOLUTION

# %%
hours = 730
monthly = ((onprem["accelerator_purchase_sar"] + onprem["server_and_installation_sar"]) / (onprem["useful_life_years"] * 12)
           + onprem["power_kw"] * hours * onprem["electricity_sar_per_kwh"] + onprem["operations_hours_per_month"] * onprem["engineer_sar_per_hour"])
try:
    assert abs(cost_per_thousand(6000, 30000) - 200) < 1e-9
except NotImplementedError:
    def cost_per_thousand(monthly_sar, requests_per_month):
        return monthly_sar / (requests_per_month / 1000)
    print("cost_per_thousand is not written yet: using the reference version")
expected = onprem["expected_requests_per_month"]
per_second = best["per_second"] if best else results[0]["per_second"]
at_capacity = per_second * 3600 * hours
rows = [("monthly cost of the machine, SAR", f"{monthly:,.0f}"),
        ("expected requests per month", f"{expected:,}"),
        ("that is, requests per second on average", f"{expected / (3600 * hours):.4f}"),
        ("SAR per 1,000 at the expected volume", f"{cost_per_thousand(monthly, expected):,.2f}"),
        ("measured capacity, requests per second", per_second),
        ("requests per month at that capacity", f"{at_capacity:,.0f}"),
        ("SAR per 1,000 at capacity", f"{cost_per_thousand(monthly, at_capacity):,.4f}"),
        ("share of capacity the chamber would use", f"{expected / at_capacity:.2%}")]
Q.show("Cost per thousand requests", rows)
Q.write_json(out / "load.json", {"measured_here": LIVE and results is not REF["levels"], "levels": results, "latency_target_ms": LIMIT_MS, "capacity": best,
                                 "monthly_sar": round(monthly, 2), "sar_per_1000_expected": round(cost_per_thousand(monthly, expected), 2),
                                 "sar_per_1000_capacity": round(cost_per_thousand(monthly, at_capacity), 4)})

# %% [markdown]
# 👀 **Look at** the last line. The chamber's volume is a small share of what one machine can answer:
# the cost per thousand is set by the idle machine, not by the model. That is the finding of Lab 1.4
# again, now with a measured throughput in it, and it is the strongest argument for putting a second
# and a third adapter on the same base. The throughput here is the course model's on a processor:
# replace it with your own measurement before any figure leaves the room.

# %% [markdown]
# ## ✱ Stretch · What does an adapter cost in latency?

# %%
if LIVE and results is not REF["levels"]:
    try:
        with_adapters = next(r for r in results if r["users"] == 8)
        alone = load_test(8, models="qayd-base", label="base_u8")
        print(f"8 in flight · base alone: {alone['per_second']} requests/s, p50 {alone['p50_ms']:.0f} ms · two adapters: {with_adapters['per_second']} requests/s, p50 {with_adapters['p50_ms']:.0f} ms")
        print("(The base alone answers these requests in words, with a different number of tokens: compare per token before you conclude.)")
    except Exception as error:
        print("The comparison could not run here:", type(error).__name__)
else:
    print("The stretch needs the live server.")

# %%
# Stop the server: a process you started is yours to stop.
server.terminate()
try:
    server.wait(timeout=20)
except subprocess.TimeoutExpired:
    server.kill()
print("server stopped")

# %% [markdown]
# ## Check yourself
#
# 1. Why does throughput rise with the number of requests in flight, up to a point?
# 2. A report says "the server handles 40 requests per second". What is missing?
# 3. The cost per thousand at the chamber's volume is far above the cost at capacity. What follows?
#
# <details><summary>Answers</summary>
#
# 1. Batching: several requests share each forward pass. Past the point where the batch is full, more
#    requests only wait.
# 2. The latency at that throughput (a percentile, not a mean), the request mix and length, the
#    hardware, and whether any request failed.
# 3. The machine is mostly idle. Share it (more adapters, more services), or accept the premium for
#    what it buys: data that stays on the chamber's servers.
#
# </details>
#
# **Carry forward.** `artifacts/4.3/load.json`, for the cost line of the memo.
