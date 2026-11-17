# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6", "transformers>=4.56", "peft>=0.17", "jinja2>=3.1"]
# ///
# %% [markdown]
# # Lab 3.2 · The same adapter on a 4-bit base
#
# Day 3 · Session 2 (Quantised adapters and memory) · 25 min.
#
# Jarrah Al-Suhaim has one accelerator. Whether a model fits on it is arithmetic you can do before you
# rent anything: parameters times bits. You do that arithmetic, round a block of weights to 16 levels by
# hand, train yesterday's adapter on a 4-bit base, and compare it with the 16-bit run.
#
# **Two tiers.** On a notebook GPU the base is loaded in 4 bits with bitsandbytes and memory is
# measured. On a processor the course model's weights are rounded to the same 16 levels by
# `tuning.quantise_4bit_`, and memory is computed, because the rounded values still sit in ordinary
# tensors. The effect of rounding on quality is real in both.
#
# **Reference state.** `data/tuning/train.jsonl`, `data/reference/d3_1_run.json` (the supplied 16-bit
# run) and `data/reference/d3_base.json` (the base, untrained, at both precisions).
#
# **Tasks**
# - [ ] 1 · Memory arithmetic for 16 and 4 bits (6')
# - [ ] 2 · Load the 4-bit base and score it untrained (4')
# - [ ] 3 · Train the same adapter on it (8')
# - [ ] 4 · Compare memory, time and score with the 16-bit run (7')
# - [ ] ✱ Stretch · Load your 4-bit adapter on the 16-bit base and score it: does it transfer?

# %%
#@setup
Q.banner("Lab 3.2")

# %%
import torch

import tuning as U

TIER = U.tier()
train_rows = U.rows_from(Q.load("tuning/train"))
val_rows = U.rows_from(Q.load("tuning/val"))
val = Q.load("requests_val")
if Q.QUICK:
    val = val[:32]
texts = [r["text"] for r in val]
RUN16, BASE = Q.reference("d3_1_run"), Q.reference("d3_base")
print("tier:", TIER)

# %% [markdown]
# ## 1 · Memory arithmetic for 16 and 4 bits (6')
#
# A weight stored in 16 bits takes two bytes. Stored in 4 bits it takes half a byte, plus a share of
# the **scale** that each block of 64 weights keeps (one 16-bit number per block).
#
# ✏️ **Exercise** · Complete `weights_mb`: megabytes (2**20 bytes) for `n` weights at `bits` bits each.
# For 4 bits, add one 16-bit scale for every 64 weights.

# %%
def weights_mb(n, bits):
    # BEGIN SOLUTION
    total_bits = n * bits + (n / 64 * 16 if bits == 4 else 0)
    return total_bits / 8 / 2**20
    # END SOLUTION

# %%
# Check, with a safety net
def _weights_mb(n, bits):
    return (n * bits + (n / 64 * 16 if bits == 4 else 0)) / 8 / 2**20

try:
    assert abs(weights_mb(64_000_000, 16) - _weights_mb(64_000_000, 16)) < 1e-6 and abs(weights_mb(64_000_000, 4) - _weights_mb(64_000_000, 4)) < 1e-6
except NotImplementedError:
    weights_mb = _weights_mb
    print("weights_mb is not written yet: using the reference version")
print("weights only, before activations, the adapter and the optimiser:")
print(f"{'parameters':>14}  {'16 bits':>10}  {'4 bits':>10}")
for n in (600_000_000, 1_700_000_000, 7_000_000_000, 27_000_000_000):
    print(f"{n:>14,}  {weights_mb(n, 16) / 1024:>8.1f} GB  {weights_mb(n, 4) / 1024:>8.1f} GB")

# %% [markdown]
# 👀 **Look at** the last two rows against the memory of one accelerator you can name. The weights are
# the floor, not the bill: training adds activations for every token of every example in the batch,
# gradients for the adapter and the optimiser's state. That is why the same base fits for serving and
# not for training, and why batch size and sequence length are memory settings.
#
# Now the rounding itself. A 4-bit weight is one of 16 levels between -1 and 1, times the scale of its
# block.
#
# ✏️ **Exercise** · Complete `round_block`: given a block of weights and the 16 levels, return the
# rounded block: divide by the block's largest absolute value, replace each value by the nearest level,
# multiply back.

# %%
def round_block(block, levels):
    # BEGIN SOLUTION
    scale = block.abs().max()
    nearest = (block.unsqueeze(-1) / scale - levels).abs().argmin(dim=-1)
    return levels[nearest] * scale
    # END SOLUTION

# %%
torch.manual_seed(0)
block = torch.randn(64) * 0.02
try:
    rounded = round_block(block, U.NF4_LEVELS)
except NotImplementedError:
    s = block.abs().max()
    rounded = U.NF4_LEVELS[(block.unsqueeze(-1) / s - U.NF4_LEVELS).abs().argmin(dim=-1)] * s
    print("round_block is not written yet: using the reference result")
assert len(set(rounded.tolist())) <= 16
print("first five weights:", [f"{v:+.4f}" for v in block[:5].tolist()])
print("rounded           :", [f"{v:+.4f}" for v in rounded[:5].tolist()])
print(f"distinct values in the block: {len(set(rounded.tolist()))} · mean absolute error: {float((block - rounded).abs().mean()):.5f} (the weights average {float(block.abs().mean()):.5f})")

# %% [markdown]
# ## 2 · Load the 4-bit base and score it untrained (4')

# %%
base4, tok = U.load_base(bits=4)
mb4 = U.base_memory_mb(base4)
before4 = U.grade(U.generate(base4, tok, texts), val)
print(f"4-bit base: {mb4:.2f} MB of weights ({'measured' if TIER == 'hub' else 'computed'}) · {sum(g['correct'] for g in before4)} of {len(val)} right untrained")
if TIER == "tiny":
    print(f"supplied, same model: 16 bits {BASE['16']['memory_mb']} MB, {BASE['16']['val']['right']} of {BASE['16']['val']['of']} right · 4 bits {BASE['4']['memory_mb']} MB, {BASE['4']['val']['right']} of {BASE['4']['val']['of']} right")

# %% [markdown]
# 👀 **Look at** the two memory figures. They are not in the ratio 4 to 1: the embedding table is left
# at 16 bits, as bitsandbytes leaves it, and in a model this small the embeddings are a large share.
# In a model of several billion parameters the linear layers dominate and the ratio approaches the
# arithmetic of task 1.

# %% [markdown]
# ## 3 · Train the same adapter on it (8')
#
# The base is frozen and rounded; the adapter trains in full precision on top of it. This is the QLoRA
# arrangement: gradients flow through the 4-bit weights to the adapter, and the adapter learns the task
# **and** some of what rounding lost.
#
# ✏️ **Exercise** · Complete `train_on`: add a rank-8 adapter to a base and train it with the settings
# of Lab 3.1. Return `(model, log)`.

# %%
STEPS = Q.pick(300, 30)


def train_on(base):
    # BEGIN SOLUTION
    model = U.add_adapter(base, rank=8)
    log = U.train(model, tok, train_rows, val_rows, steps=STEPS, lr=2e-3, batch=8)
    return model, log
    # END SOLUTION

# %%
try:
    model4, log4 = train_on(base4)
except NotImplementedError:
    model4 = U.add_adapter(base4, rank=8)
    log4 = U.train(model4, tok, train_rows, val_rows, steps=STEPS, lr=2e-3, batch=8)
    print("train_on is not written yet: ran with the course's settings")
after4 = U.grade(U.generate(model4, tok, texts), val)
log4.update({"bits": 4, "base_memory_mb": round(mb4, 3), "val": {"right": sum(g["correct"] for g in after4), "of": len(val)}})
print(f"{log4['steps']} steps in {log4['seconds']:.0f} s · final validation loss {log4['curve'][-1]['val_loss']} · {log4['val']['right']} of {len(val)} right")

# %% [markdown]
# ## 4 · Compare memory, time and score with the 16-bit run (7')

# %%
rows = [("", "16-bit base", "4-bit base")]
if TIER == "tiny":
    rows += [("base weights, MB", RUN16["base_memory_mb"], log4["base_memory_mb"]),
             ("training time, s", RUN16["seconds"], log4["seconds"]),
             ("final validation loss", RUN16["curve"][-1]["val_loss"], log4["curve"][-1]["val_loss"]),
             ("validation requests right", f"{RUN16['val']['right']} of {RUN16['val']['of']}", f"{log4['val']['right']} of {log4['val']['of']}")]
else:
    rows += [("base weights, MB", "run Lab 3.1 on this runtime", log4["base_memory_mb"]),
             ("peak GPU memory in training, MB", "see artifacts/3.1/adapter/run_log.json", log4["peak_gpu_mb"]),
             ("validation requests right", "", f"{log4['val']['right']} of {log4['val']['of']}")]
for r in rows:
    print(f"{str(r[0]):<34}{str(r[1]):>16}{str(r[2]):>16}")
out = Q.artifacts("3.2")
U.save_adapter(model4, out / "adapter", log4)
print("saved", (out / "adapter").relative_to(Q.LABS))

# %% [markdown]
# 👀 **Look at** three things. Memory: the reason to do this at all. Time: on a GPU a 4-bit base trains
# somewhat slower, because every forward pass converts the weights back; on a processor here nothing is
# packed, so the time is the same. Score: the difference between the two columns is a few requests in
# 120, inside the noise of one run. "4-bit costs nothing" and "4-bit costs two points" are both claims
# that need the paired comparison of Day 4.

# %% [markdown]
# ## ✱ Stretch · Does the adapter transfer?
#
# Your adapter was trained on the rounded base. Load it on the 16-bit base and score it.

# %%
base16, _ = U.load_base(bits=16)
moved = U.load_adapter(base16, out / "adapter")
moved_graded = U.grade(U.generate(moved, tok, texts), val)
print(f"4-bit adapter on the 16-bit base: {sum(g['correct'] for g in moved_graded)} of {len(val)} · on its own base: {log4['val']['right']} of {len(val)}")
print("An adapter is trained against one set of base weights. Serve it on the base, and the precision, it was trained on, or measure again.")

# %% [markdown]
# ## Check yourself
#
# 1. A base has 7 billion parameters. Roughly how much memory do its weights take at 16 and at 4 bits?
# 2. The 4-bit weights fit, and training still runs out of memory. Name two settings to lower.
# 3. Why is the adapter itself not quantised?
#
# <details><summary>Answers</summary>
#
# 1. About 13 GB and about 3.5 GB.
# 2. Batch size and maximum sequence length (activations grow with both); gradient accumulation keeps
#    the effective batch.
# 3. It is tiny, so there is nothing to save, and it is what the gradient updates: it needs full
#    precision to move by small steps.
#
# </details>
#
# **Carry forward.** `artifacts/3.2/adapter/`. The supplied copy is `adapters/registry-4bit/`: Day 4
# evaluates and serves that one.
