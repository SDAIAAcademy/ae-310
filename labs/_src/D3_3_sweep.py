# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6", "transformers>=4.56", "peft>=0.17", "jinja2>=3.1"]
# ///
# %% [markdown]
# # Lab 3.3 · Read a sweep, add a run
#
# Day 3 · Session 3 (Settings that matter) · 25 min.
#
# A sweep is a table of runs that differ in one or two settings. Nobody has time to run one in a
# classroom, so eight runs are supplied: three ranks by three learning rates, with one cell missing.
# You read the table, predict the missing cell, run it, and choose a setting with a reason you could
# defend to Maysoon Al-Dabbagh.
#
# **Reference state.** `data/reference/d3_3_sweep.json`: eight runs on the course model (300 steps,
# batch 8, the supplied tuning set), each with its curve and its score on the 120 validation requests.
# On the hub tier the supplied sweep is still the course model's: read it for the pattern, and run the
# missing setting on your own base.
#
# **Tasks**
# - [ ] 1 · Read the supplied sweep (6')
# - [ ] 2 · Predict the missing setting (3')
# - [ ] 3 · Run it (9')
# - [ ] 4 · Choose on the validation set and say why (7')
# - [ ] ✱ Stretch · Plot validation accuracy against trainable parameters

# %%
#@setup
Q.banner("Lab 3.3")

# %%
import tuning as U

TIER = U.tier()
SWEEP = Q.reference("d3_3_sweep")
runs = SWEEP["runs"]
ranks = sorted({r["rank"] for r in runs} | {SWEEP["held_out"]["rank"]})
rates = sorted({r["lr"] for r in runs} | {SWEEP["held_out"]["lr"]})
MISSING = (SWEEP["held_out"]["rank"], SWEEP["held_out"]["lr"])
print(len(runs), "supplied runs · ranks", ranks, "· learning rates", rates, "· missing:", MISSING)

# %% [markdown]
# ## 1 · Read the supplied sweep (6')
#
# ✏️ **Exercise** · Complete `cell`: return the text for one cell of the table, the number of
# validation requests a run got right, or `"?"` when the sweep has no run for that rank and rate.

# %%
def cell(runs, rank, lr):
    # BEGIN SOLUTION
    for r in runs:
        if r["rank"] == rank and r["lr"] == lr:
            return str(r["val"]["right"])
    return "?"
    # END SOLUTION

# %%
# Check, with a safety net
def _cell(runs, rank, lr):
    return next((str(r["val"]["right"]) for r in runs if r["rank"] == rank and r["lr"] == lr), "?")

try:
    assert all(cell(runs, a, b) == _cell(runs, a, b) for a in ranks for b in rates)
except NotImplementedError:
    cell = _cell
    print("cell is not written yet: using the reference version")


def table(runs):
    print("validation requests right, of", runs[0]["val"]["of"])
    print(f"{'rank':>6}" + "".join(f"{'lr ' + format(lr, 'g'):>12}" for lr in rates) + f"{'trainable':>12}")
    for rank in ranks:
        trainable = next(r["trainable"] for r in runs + [SWEEP["held_out"]] if r["rank"] == rank)
        print(f"{rank:>6}" + "".join(f"{cell(runs, rank, lr):>12}" for lr in rates) + f"{trainable:>12,}")


table(runs)
print()
print("final validation loss:")
for r in sorted(runs, key=lambda r: (r["rank"], r["lr"])):
    low = min(p["val_loss"] for p in r["curve"])
    print(f"  rank {r['rank']:>2}  lr {r['lr']:<7g} end {r['curve'][-1]['val_loss']:.4f}  lowest {low:.4f}  {r['seconds']:.0f} s")

# %% [markdown]
# 👀 **Look at** the table by column, then by row. Which setting moves the score more: the learning
# rate or the rank? Where is the validation loss at the end above its lowest point?

# %% [markdown]
# ## 2 · Predict the missing setting (3')
#
# ✏️ **Exercise** · Before running anything, write your prediction for the missing cell: a number of
# validation requests right, and one sentence of reasoning from its neighbours in the table.

# %%
prediction = None      # ✏️ your number, for example 97
reasoning = ""         # ✏️ one sentence
print("prediction for rank", MISSING[0], "at learning rate", MISSING[1], ":", prediction, "·", reasoning)

# %% [markdown]
# ## 3 · Run it (9')
#
# ✏️ **Exercise** · Complete `run_setting`: train an adapter of the given rank at the given learning
# rate with the sweep's other settings, score it on the validation requests, and return a run in the
# sweep's format.

# %%
train_rows = U.rows_from(Q.load("tuning/train"))
val_rows = U.rows_from(Q.load("tuning/val"))
val = Q.load("requests_val")
STEPS = Q.pick(SWEEP["steps"], 30)


def run_setting(rank, lr):
    base, tok = U.load_base()
    # BEGIN SOLUTION
    model = U.add_adapter(base, rank=rank)
    log = U.train(model, tok, train_rows, val_rows, steps=STEPS, lr=lr, batch=SWEEP["batch"])
    graded = U.grade(U.generate(model, tok, [r["text"] for r in val]), val)
    return {"rank": rank, "lr": lr, "trainable": log["trainable"], "seconds": log["seconds"], "curve": log["curve"],
            "val": {"right": sum(g["correct"] for g in graded), "of": len(val)}}
    # END SOLUTION

# %%
if Q.QUICK:
    mine = dict(SWEEP["held_out"])
    print("quick mode: using the supplied result for the missing run")
else:
    try:
        mine = run_setting(*MISSING)
    except NotImplementedError:
        mine = dict(SWEEP["held_out"])
        print("run_setting is not written yet: using the supplied result for the missing run")
print(f"rank {mine['rank']} at {mine['lr']}: {mine['val']['right']} of {mine['val']['of']} right · end loss {mine['curve'][-1]['val_loss']} · {mine['seconds']:.0f} s")
print("your prediction was:", prediction)
if TIER == "tiny":
    print("the course's run of the same setting:", SWEEP["held_out"]["val"]["right"], "of", SWEEP["held_out"]["val"]["of"])

# %% [markdown]
# ## 4 · Choose on the validation set and say why (7')
#
# The choice is made on **validation** requests. The test set stays closed until Day 4: a setting chosen
# on the test set has seen the test set.
#
# ✏️ **Exercise** · Complete `choose`: return the run to ship. Rule: the highest validation score; among
# runs within 4 requests of the best, the one with the fewest trainable parameters.

# %%
def choose(runs):
    # BEGIN SOLUTION
    best = max(r["val"]["right"] for r in runs)
    near = [r for r in runs if best - r["val"]["right"] <= 4]
    return min(near, key=lambda r: (r["trainable"], -r["val"]["right"]))
    # END SOLUTION

# %%
full = runs + [mine] if TIER == "tiny" else runs + [SWEEP["held_out"]]
try:
    chosen = choose(full)
except NotImplementedError:
    top = max(r["val"]["right"] for r in full)
    chosen = min([r for r in full if top - r["val"]["right"] <= 4], key=lambda r: (r["trainable"], -r["val"]["right"]))
    print("choose is not written yet: using the reference choice")
table(full)
print(f"\nchosen: rank {chosen['rank']} at learning rate {chosen['lr']} · {chosen['val']['right']} of {chosen['val']['of']} · {chosen['trainable']:,} trainable parameters")
Q.write_json(Q.artifacts("3.3") / "choice.json", {"rank": chosen["rank"], "lr": chosen["lr"], "val": chosen["val"], "rule": "highest validation score; within 4 requests, the fewest trainable parameters", "my_run": mine})

# %% [markdown]
# 👀 **Look at** the rule, not only the result. "Within 4 requests" says that a difference of four in
# 120 is not a difference: one run per cell, one seed. The margin is not a guess: your run of the missing
# cell and the course's run of the same cell, with the same seed, differ by a few requests, because the
# order of the arithmetic differs between machines. A sweep read to the last digit chooses noise. The
# honest version of this table has three seeds per cell, and the capstone asks you to say when yours
# does not.

# %% [markdown]
# ## ✱ Stretch · Score against trainable parameters

# %%
for r in sorted(full, key=lambda r: (r["trainable"], r["lr"])):
    print(f"{r['trainable']:>8,}  lr {r['lr']:<7g} {'#' * round(40 * r['val']['right'] / r['val']['of'])} {r['val']['right']}")
print("More trainable parameters is a cost (memory, time, a larger file per adapter), not a quality setting.")

# %% [markdown]
# ## Check yourself
#
# 1. Two cells differ by one request in 120. Which do you choose?
# 2. Why is the setting chosen on validation requests and not on the test set?
# 3. A colleague wants to sweep six settings at five values each. What do you say?
#
# <details><summary>Answers</summary>
#
# 1. Neither on that evidence: take the cheaper one (lower rank), or run more seeds.
# 2. Choosing is fitting. A test set used to choose a setting no longer estimates how the chosen model
#    does on new requests.
# 3. That is 15,625 runs. Fix what the literature and one small sweep already tell you (targets,
#    schedule, batch), sweep the learning rate first, then the rank, and spend the rest on data.
#
# </details>
#
# **Carry forward.** `artifacts/3.3/choice.json`: the setting, and the rule that chose it, for the model
# card.
