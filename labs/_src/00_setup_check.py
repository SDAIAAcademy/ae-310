# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Setup check
#
# Run this notebook once before Day 1. It imports every library the labs use, reads one data file, asks
# each hosted model one question (or says that the scripted stand-in will answer), and loads the course's
# own small model. Every line that prints "missing" tells you what to install or set.

# %%
#@setup
Q.banner("Setup check")

# %%
from importlib import metadata

# Versions are read from the installed packages without importing them: Locust, for one, changes
# how the network works when it is imported, which a notebook must not do to itself.
missing = []
for name in ("numpy", "matplotlib", "torch", "transformers", "peft", "tokenizers", "openai", "httpx", "fastapi", "uvicorn", "locust"):
    try:
        print(f"ok       {name} {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        missing.append(name)
        print(f"missing  {name}")
print()
print("data:", len(Q.load("requests_100")), "requests in data/requests_100.jsonl")

# %%
import prompts

for role in ("small", "large"):
    reply = Q.chat([{"role": "system", "content": prompts.FROZEN}, {"role": "user", "content": Q.load("requests_100")[0]["text"]}], role)
    print(f"{role:5s} model answered from: {reply['source']:8s} {reply['text'][:80]!r}")
if Q.using_standin():
    print("\nNo working key: the scripted stand-in answers in Days 1 and 2. To use the hosted models, put OPENROUTER_API_KEY")
    print("in labs/.env (locally) or in the Secrets panel (Colab) and run this notebook again.")

# %%
import torch

import tuning as U

print("tier:", U.tier(), "· device:", U.device())
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}, {gpu.total_memory / 2**30:.1f} GB")
else:
    print("No GPU: Days 3 and 4 train the course's own small model on this processor. That is a supported way to take the course.")
model, tok = U.load_base("tiny")
answer = U.generate(model, tok, ["Translate to Arabic: Riyadh"], which="tiny")[0]
print("course model:", U.count_parameters(model)["total"], "parameters; it answers:", answer)
assert answer.strip() == "الرياض", "the course model did not load correctly"
print()
print("All set." if not missing else f"Install what is missing: {missing}")
