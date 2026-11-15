"""Training, generation and scoring for the adapter labs of Days 3 and 4.

One code path, two tiers of base model:

- "tiny": the course's own small decoder in models/qayd-tiny (about 2 million parameters, the
  Llama layout, its own tokeniser and chat template). It trains on a laptop processor in about
  a minute, so every lab runs anywhere.
- "hub": a small open-weight model from the Hugging Face Hub (ADAPT_BASE), for a free notebook
  GPU. It is downloaded into the notebook runtime, never onto a laptop.

The training loop is plain PyTorch on purpose: every line of it is on a slide.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

import chamber as C

LABS = Path(__file__).resolve().parent
TINY_DIR = LABS / "models" / "qayd-tiny"
HUB_BASE = os.environ.get("ADAPT_BASE", "Qwen/Qwen3-0.6B")

SYSTEM_TINY = "You are Qayd, the registry assistant."
TARGETS = ["q_proj", "v_proj"]

if not torch.cuda.is_available():  # a small model on a processor gains nothing from many threads
    torch.set_num_threads(int(os.environ.get("ADAPT_THREADS", "4")))

# The 16 levels of the 4-bit NormalFloat type of the QLoRA paper (Dettmers et al., 2023, appendix E)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])


def tier() -> str:
    """"hub" on a GPU unless ADAPT_TIER says otherwise; "tiny" on a processor."""
    chosen = os.environ.get("ADAPT_TIER", "").strip().lower()
    if chosen in ("tiny", "hub"):
        return chosen
    return "hub" if torch.cuda.is_available() else "tiny"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed(n: int = 310) -> None:
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def load_tokenizer(which: str | None = None):
    from transformers import AutoTokenizer

    which = which or tier()
    tok = AutoTokenizer.from_pretrained(str(TINY_DIR) if which == "tiny" else HUB_BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generation in batches reads the last position
    return tok


def quantise_4bit_(model, block: int = 64) -> int:
    """Round every linear weight of a model to 16 levels per block of 64 values, in place.

    This is what a 4-bit base holds: each block keeps one scale (its largest absolute value)
    and a 4-bit index per weight. bitsandbytes does the same on a GPU and stores the indices
    packed; here the rounded values stay in ordinary tensors, so the arithmetic of the labs is
    exact and the memory figure is computed, not measured. Returns the number of weights rounded.
    """
    n = 0
    levels = NF4_LEVELS
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "lora_" not in name and "lm_head" not in name:
                w = module.weight
                flat = w.reshape(-1)
                pad = (-flat.numel()) % block
                blocks = torch.nn.functional.pad(flat, (0, pad)).reshape(-1, block)
                scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
                idx = ((blocks / scale).unsqueeze(-1) - levels).abs().argmin(dim=-1)
                rounded = (levels[idx] * scale).reshape(-1)[: flat.numel()].reshape(w.shape)
                module.weight.copy_(rounded)
                n += flat.numel()
    return n


def load_base(which: str | None = None, bits: int = 16):
    """The frozen base model and its tokeniser. bits is 16 or 4."""
    from transformers import AutoModelForCausalLM

    which = which or tier()
    tok = load_tokenizer(which)
    if which == "tiny":
        model = AutoModelForCausalLM.from_pretrained(str(TINY_DIR))
        if bits == 4:
            quantise_4bit_(model)
        model.to(device())
    elif bits == 4:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(HUB_BASE, quantization_config=quant, device_map={"": 0})
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(HUB_BASE, dtype=dtype).to(device())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.adapt_tier, model.adapt_bits = which, bits
    return model, tok


def base_memory_mb(model) -> float:
    """Megabytes the base weights take at the precision they were loaded in.

    A GPU model reports what it holds. The course model is computed: 16 bits per weight, or for
    a 4-bit base 4 bits per linear weight plus one 16-bit scale per block of 64, with the
    embeddings left at 16 bits (as bitsandbytes leaves them).
    """
    if getattr(model, "adapt_tier", "tiny") == "hub":
        return model.get_memory_footprint() / 2**20
    linear = sum(m.weight.numel() for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lora_" not in n and "lm_head" not in n)
    total = sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n)
    if getattr(model, "adapt_bits", 16) == 4:
        bits = linear * 4 + (linear / 64) * 16 + (total - linear) * 16
    else:
        bits = total * 16
    return bits / 8 / 2**20


_NAMES: list[str] | None = None


def known_names() -> list[str]:
    """The owners' names of the registry: the list the masking step looks for."""
    global _NAMES
    if _NAMES is None:
        rows = json.loads((LABS / "data" / "registry.json").read_text(encoding="utf-8"))
        _NAMES = sorted({r[k] for r in rows for k in ("owner_en", "owner_ar")})
    return _NAMES


def masked(text: str) -> str:
    """A request as the model is allowed to see it, in training and in service: personal data masked."""
    return C.mask_personal_data(text, known_names())


def messages_for(text: str, tool_call: dict | None = None, which: str | None = None) -> list[dict]:
    """The conversation for one request; with a tool call, the assistant turn that answers it."""
    which = which or tier()
    text = masked(text)
    system = SYSTEM_TINY if which == "tiny" else hub_system_prompt()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    if tool_call is not None:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": tool_call}]})
    return msgs


def hub_system_prompt() -> str:
    """The frozen instruction of the hub tier: it carries the code table, which is knowledge."""
    codes = "; ".join(f"{a[0]} {a[1]}" for a in C.ACTIVITIES)
    return (SYSTEM_TINY + " Answer every request with exactly one tool call. Write registration numbers with Western digits. "
            "Copy the company name as written. Cities in English. Escalate when the registration number is missing or the request is outside the registry. "
            "Activity codes: " + codes + ".")


def render(tok, text: str, tool_call: dict | None = None, which: str | None = None) -> str:
    """The exact text the model reads (and, with a tool call, the text it is trained to continue)."""
    which = which or tier()
    msgs = messages_for(text, tool_call, which)
    kwargs = {} if which == "tiny" else {"tools": C.TOOLS, "enable_thinking": False}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=tool_call is None, **kwargs)


def encode(tok, rows: list[dict], max_len: int = 0, which: str | None = None):
    """Token ids and labels for training. The loss is taken on the assistant turn only.

    rows: {"text", "call"}. Labels are -100 on the prompt and the padding, so the model is not
    trained to write the request, only to answer it.
    """
    which = which or tier()
    max_len = max_len or (224 if which == "tiny" else 1024)
    ids, labels = [], []
    for row in rows:
        prompt = tok(render(tok, row["text"], None, which), add_special_tokens=False)["input_ids"]
        full = tok(render(tok, row["text"], row["call"], which), add_special_tokens=False)["input_ids"]
        full = full[:max_len]
        lab = [-100] * min(len(prompt), len(full)) + full[len(prompt):]
        ids.append(full)
        labels.append(lab)
    width = max(len(x) for x in ids)
    pad = tok.pad_token_id
    x = torch.full((len(ids), width), pad, dtype=torch.long)
    y = torch.full((len(ids), width), -100, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for i, (a, b) in enumerate(zip(ids, labels)):  # right padding for training
        x[i, : len(a)] = torch.tensor(a)
        y[i, : len(b)] = torch.tensor(b)
        mask[i, : len(a)] = 1
    return x, y, mask


def add_adapter(model, rank: int = 8, alpha: int | None = None, targets: list[str] | None = None, dropout: float = 0.0, name: str = "default"):
    """Wrap a frozen base with a low-rank adapter (PEFT)."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, target_modules=targets or TARGETS, lora_dropout=dropout, task_type="CAUSAL_LM")
    return get_peft_model(model, config, adapter_name=name)


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "share": trainable / total}


@torch.no_grad()
def loss_on(model, batch_tensors, batch: int = 32) -> float:
    x, y, mask = batch_tensors
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(x), batch):
        xb, yb, mb = (t[i : i + batch].to(device()) for t in (x, y, mask))
        out = model(input_ids=xb, attention_mask=mb, labels=yb)
        k = int((yb != -100).sum())
        total += float(out.loss) * k
        n += k
    return total / max(n, 1)


def train(model, tok, train_rows, val_rows, steps: int = 300, lr: float = 2e-3, batch: int = 16, warmup: float = 0.06,
          schedule: str = "cosine", log_every: int = 25, seed_n: int = 310, which: str | None = None) -> dict:
    """Train the trainable parameters of a model (the adapter) and return the run log.

    The log holds what a run must leave behind: the settings, the loss on the training batches
    and on the validation set every log_every steps, seconds, and peak GPU memory when there is
    a GPU.
    """
    which = which or getattr(model, "adapt_tier", None) or tier()
    seed(seed_n)
    xt, yt, mt = encode(tok, train_rows, which=which)
    val = encode(tok, val_rows, which=which)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, int(steps * warmup))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        if schedule == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator().manual_seed(seed_n)
    order = torch.randperm(len(xt), generator=gen)
    cursor, curve, running, started = 0, [], [], time.perf_counter()
    for step in range(steps):
        if cursor + batch > len(order):
            order, cursor = torch.randperm(len(xt), generator=gen), 0
        idx = order[cursor : cursor + batch]
        cursor += batch
        width = int(mt[idx].sum(dim=1).max())
        xb, yb, mb = xt[idx, :width].to(device()), yt[idx, :width].to(device()), mt[idx, :width].to(device())
        for group in opt.param_groups:
            group["lr"] = lr * factor(step)
        model.train()
        loss = model(input_ids=xb, attention_mask=mb, labels=yb).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        running.append(float(loss))
        if (step + 1) % log_every == 0 or step + 1 == steps:
            curve.append({"step": step + 1, "train_loss": round(sum(running) / len(running), 4), "val_loss": round(loss_on(model, val), 4)})
            running = []
    seconds = time.perf_counter() - started
    model.eval()
    return {
        "tier": which, "steps": steps, "lr": lr, "batch": batch, "schedule": schedule, "seed": seed_n,
        "train_rows": len(train_rows), "val_rows": len(val_rows), "trainable": count_parameters(model)["trainable"],
        "seconds": round(seconds, 1), "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
        "curve": curve,
    }


@torch.no_grad()
def generate(model, tok, texts: list[str], max_new_tokens: int = 0, batch: int = 32, which: str | None = None) -> list[str]:
    """Greedy answers to a list of requests, as text up to the end-of-turn token."""
    which = which or getattr(model, "adapt_tier", None) or tier()
    max_new_tokens = max_new_tokens or (72 if which == "tiny" else 128)
    model.eval()
    outputs = []
    end = tok.convert_tokens_to_ids("<|im_end|>")
    stops = [e for e in {tok.eos_token_id, end} if isinstance(e, int) and e >= 0]
    for i in range(0, len(texts), batch):
        prompts = [render(tok, t, None, which) for t in texts[i : i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device())
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            text = tok.decode(row, skip_special_tokens=False)
            for marker in ("<|im_end|>", tok.pad_token or "<|pad|>", tok.eos_token or "<|im_end|>"):
                text = text.split(marker)[0]
            outputs.append(text.strip())
    return outputs


def grade(outputs: list[str], rows: list[dict]) -> list[dict]:
    """One line per request: whether the output is a valid call and whether it is the right one."""
    graded = []
    for out, row in zip(outputs, rows):
        call = C.parse_call(out)
        graded.append({"id": row["id"], "lang": row.get("lang"), "output": out, "valid": call is not None and not C.schema_errors(call),
                       "correct": C.same_call(call, row["gold"])})
    return graded


def accuracy(graded: list[dict], lang: str | None = None) -> float:
    rows = [g for g in graded if lang is None or g["lang"] == lang]
    return sum(g["correct"] for g in rows) / max(len(rows), 1)


def save_adapter(model, folder: Path, log: dict | None = None) -> None:
    folder = Path(folder)
    model.save_pretrained(str(folder))
    (folder / "README.md").unlink(missing_ok=True)  # PEFT writes a card template; the course's cards are written in Lab 4.4
    if log is not None:
        (folder / "run_log.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8", newline="\n")


def load_adapter(base, folder: Path, name: str = "default"):
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(folder), adapter_name=name)
    model.eval()
    model.adapt_tier, model.adapt_bits = getattr(base, "adapt_tier", "tiny"), getattr(base, "adapt_bits", 16)
    return model
