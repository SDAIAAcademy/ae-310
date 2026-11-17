# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6", "transformers>=4.56", "peft>=0.17", "jinja2>=3.1"]
# ///
# %% [markdown]
# # Lab 3.1 · Train an adapter, read the curves
#
# Day 3 · Session 1 (Low-rank adapters) · 25 min.
#
# Yesterday's 661 examples meet a model. You freeze the base, add a low-rank adapter to two of its
# weight matrices, train only the adapter, and read what the run tells you: two loss curves, and the
# score on validation requests before and after.
#
# **Two tiers, one code path.** On a notebook GPU (Colab: Runtime → Change runtime type → GPU) the lab
# trains a small open-weight model from the Hub. On a processor it trains the course's own small model
# in `models/qayd-tiny/`, with the same code, in a few minutes. The banner says which you have.
#
# **Reference state.** `data/tuning/train.jsonl` and `data/tuning/val.jsonl` (supplied, cleaned),
# `data/requests_val.jsonl` for scoring, and `data/reference/d3_1_run.json`, the log of the course's own
# run on the course model. Nothing of your own from Day 2 is needed.
#
# **Tasks**
# - [ ] 1 · Load the base and count what trains (4')
# - [ ] 2 · Score the base on validation requests (4')
# - [ ] 3 · Train the adapter (8')
# - [ ] 4 · Read the two loss curves (5')
# - [ ] 5 · Score the adapter and save it with its log (4')
# - [ ] ✱ Stretch · Read the supplied log of a run that went wrong and name the setting

# %%
#@setup
Q.banner("Lab 3.1")

# %%
import tuning as U

TIER = U.tier()
print("tier:", TIER, "· device:", U.device(), "· base:", "models/qayd-tiny" if TIER == "tiny" else U.HUB_BASE)
train_rows = U.rows_from(Q.load("tuning/train"))
val_rows = U.rows_from(Q.load("tuning/val"))
val = Q.load("requests_val")
if Q.QUICK:
    val = val[:32]
REF = Q.reference("d3_1_run")
print(len(train_rows), "training examples,", len(val_rows), "validation examples")

# %% [markdown]
# ## 1 · Load the base and count what trains (4')
#
# `U.load_base()` returns the base with every parameter frozen. `U.add_adapter(base, rank=8)` wraps the
# query and value projections of every attention layer with a low-rank pair: for a weight matrix `W` of
# shape `d_out × d_in`, the adapter adds `B·A`, where `A` is `rank × d_in` and `B` is `d_out × rank`.
# Only `A` and `B` train.
#
# ✏️ **Exercise** · Complete `adapter_parameters`: how many parameters does a rank-`r` adapter add to
# **one** weight matrix of shape `d_out × d_in`?

# %%
def adapter_parameters(d_out, d_in, rank):
    # BEGIN SOLUTION
    return rank * d_in + d_out * rank
    # END SOLUTION

# %%
# Check, with a safety net
base, tok = U.load_base()
model = U.add_adapter(base, rank=8)
counted = U.count_parameters(model)
wrapped = [(n, m.weight.shape) for n, m in base.named_modules() if n.split(".")[-1] in U.TARGETS and hasattr(m, "weight")]
try:
    mine = sum(adapter_parameters(shape[0], shape[1], 8) for _, shape in wrapped)
    assert mine == counted["trainable"], (mine, counted["trainable"])
except NotImplementedError:
    mine = counted["trainable"]
    print("adapter_parameters is not written yet: using the count from the model")
Q.show("What trains", [("weight matrices wrapped", len(wrapped)), ("shape of the first", tuple(wrapped[0][1])),
                       ("trainable parameters", f"{counted['trainable']:,}"), ("all parameters", f"{counted['total']:,}"),
                       ("share that trains", f"{counted['share']:.2%}")])

# %% [markdown]
# ## 2 · Score the base on validation requests (4')
#
# Before any training: what does the base do with the chamber's requests? `U.generate` renders each
# request with the model's chat template and decodes greedily; `U.grade` parses the output and compares
# it with the right call.

# %%
texts = [r["text"] for r in val]
before = U.grade(U.generate(model, tok, texts), val)  # an untrained adapter adds exactly zero: this is the base
print(f"base: {sum(g['correct'] for g in before)} of {len(val)} right, {sum(g['valid'] for g in before)} valid calls")
for g in before[:3]:
    print("  ", g["output"].replace("\n", " ")[:110])

# %% [markdown]
# 👀 **Look at** the three outputs. Is the base failing at knowledge, format or behaviour (Lab 1.1)?

# %% [markdown]
# ## 3 · Train the adapter (8')
#
# The loop inside `U.train` is plain PyTorch, about twenty lines: sample a batch, render it with the
# template, take the loss on the assistant's turn only (yesterday's label mask), step the optimiser,
# and every 25 steps measure the loss on the validation examples.
#
# ✏️ **Exercise** · Complete `start_run`: call `U.train(model, tok, train_rows, val_rows, ...)` with the
# course's settings (300 steps, a learning rate of `2e-3`, a batch of 8) and return its log. An adapter
# starts from zero and takes a much higher learning rate than full fine-tuning would.

# %%
STEPS = Q.pick(300, 30)


def start_run():
    # BEGIN SOLUTION
    return U.train(model, tok, train_rows, val_rows, steps=STEPS, lr=2e-3, batch=8)
    # END SOLUTION

# %%
try:
    log = start_run()
except NotImplementedError:
    log = U.train(model, tok, train_rows, val_rows, steps=STEPS, lr=2e-3, batch=8)
    print("start_run is not written yet: ran with the course's settings")
print(f"{log['steps']} steps in {log['seconds']:.0f} s · peak GPU memory: {log['peak_gpu_mb']} MB")

# %% [markdown]
# ## 4 · Read the two loss curves (5')
#
# ✏️ **Exercise** · Complete `read_curve`: from the list of `{"step", "train_loss", "val_loss"}`
# points, return the step with the lowest validation loss and whether the validation loss at the end is
# more than 5% above that lowest value (a sign that the run went on too long).

# %%
def read_curve(curve):
    # BEGIN SOLUTION
    best = min(curve, key=lambda p: p["val_loss"])
    return best["step"], curve[-1]["val_loss"] > 1.05 * best["val_loss"]
    # END SOLUTION

# %%
try:
    best_step, too_long = read_curve(log["curve"])
except NotImplementedError:
    b = min(log["curve"], key=lambda p: p["val_loss"])
    best_step, too_long = b["step"], log["curve"][-1]["val_loss"] > 1.05 * b["val_loss"]
    print("read_curve is not written yet: using the reference reading")
print("step   train    val")
for p in log["curve"]:
    print(f"{p['step']:4d}  {p['train_loss']:.4f}  {p['val_loss']:.4f}  " + "#" * int(40 * min(1.0, p["val_loss"] / log["curve"][0]["val_loss"])))
print("lowest validation loss at step", best_step, "· ran too long:", too_long)
if TIER == "tiny" and not Q.QUICK:
    print("the course's run:", REF["curve"][-1], "in", REF["seconds"], "s")

# %% [markdown]
# 👀 **Look at** the gap between the two columns. Training loss is measured on batches the adapter has
# just been fitted to; validation loss on examples it never trains on. Both falling together: the
# adapter is learning the task. Training falling while validation rises: it is memorising the set.

# %% [markdown]
# ## 5 · Score the adapter and save it with its log (4')
#
# A loss is not a score. The registry accepts a call or it does not, so the adapter is graded like the
# base was: exact match on the validation requests.

# %%
after = U.grade(U.generate(model, tok, texts), val)
right_before, right_after = sum(g["correct"] for g in before), sum(g["correct"] for g in after)
log["val"] = {"right": right_after, "of": len(val)}
Q.show("Validation requests", [("base, right", f"{right_before} of {len(val)}"), ("adapter, right", f"{right_after} of {len(val)}"),
                               ("adapter, valid calls", sum(g["valid"] for g in after))])
out = Q.artifacts("3.1")
U.save_adapter(model, out / "adapter", log)
size = sum(f.stat().st_size for f in (out / "adapter").iterdir()) / 1024
print(f"saved {(out / 'adapter').relative_to(Q.LABS)}: {size:.0f} KB, with run_log.json")
if TIER == "tiny" and not Q.QUICK:
    print(f"the course's run scored {REF['val']['right']} of {REF['val']['of']}; a difference of a few requests between two runs is noise")
    assert right_after > right_before, "the adapter should beat the base on validation requests"

# %% [markdown]
# 👀 **Look at** the size of the saved folder. The adapter is the only thing you trained and the only
# thing you ship: the base stays as it was, which is what lets one base serve several adapters (Lab 4.3).
# The run log beside it holds the settings, the seed, the curves and the time. An adapter without its
# log cannot be reproduced.

# %% [markdown]
# ## ✱ Stretch · A run that went wrong
#
# `data/reference/d3_1_bad_run.json` is the log of another run on the same data. Read its curve and its
# settings and name what was wrong.

# %%
bad = Q.reference("d3_1_bad_run")
print({k: bad[k] for k in ("steps", "lr", "schedule", "batch")})
for p in bad["curve"]:
    print(f"{p['step']:4d}  {p['train_loss']:.4f}  {p['val_loss']:.4f}")
print("validation requests right:", bad["val"]["right"], "of", bad["val"]["of"])

# %% [markdown]
# ## Check yourself
#
# 1. Why does the adapter take a learning rate that would wreck a full fine-tuning run?
# 2. The training loss is 0.02 and the validation loss 0.30 and rising. What do you change first?
# 3. What must be saved with an adapter for someone else to reproduce it?
#
# <details><summary>Answers</summary>
#
# 1. Only the small matrices `A` and `B` move, `B` starts at zero, and the base is frozen: the update
#    is low-rank and scaled, so a step that size cannot disturb what the base knows.
# 2. Stop earlier (fewer steps), then look at the data: a set with many near-copies makes memorising
#    easy. More rank would make it worse.
# 3. The name and revision of the base, the precision it was loaded in, the tuning set, the settings,
#    the seed and the curves: the run log, and the dataset card.
#
# </details>
#
# **Carry forward.** `artifacts/3.1/adapter/`. The supplied copy of the course's run is
# `adapters/registry-16bit/`.
