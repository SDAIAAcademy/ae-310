"""Pretrain the course's small decoder, models/qayd-tiny, and train its tokeniser.

    uv run python _tools/pretrain_base.py

Run once by the course authors, followed by `pretrain_base.py more 6000` (a second pass on
fresh pairs) and `pretrain_base.py more 2500 2 0.15` (a third, in which 15% of the requests
that must be escalated are answered with the escalation call, so that the model has produced
every token of the tool interface at least sometimes); the result is committed. The model stands in for "a small
open-weight model" wherever a lab must run on a processor: it has general abilities (copying,
a small dictionary, comparing and counting, naming a language), it refuses requests for
personal data, and it has seen registration requests mostly in an older answer format, so
that prompted as it is, it fails at the chamber's tool calls. Seed 3100.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

import chamber as C  # noqa: E402
import prepare_data as P  # noqa: E402
import tuning as U  # noqa: E402

SEED = 3100
OUT = U.TINY_DIR

CHAT_TEMPLATE = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m['role'] + '\\n' -}}"
    "{%- if m['role'] == 'assistant' and m['tool_calls'] is defined and m['tool_calls'] -%}"
    "{%- for tc in m['tool_calls'] -%}"
    "{{- '<tool_call>\\n' + (tc['function'] | tojson) + '\\n</tool_call>' -}}"
    "{%- endfor -%}"
    "{%- else -%}"
    "{{- m['content'] -}}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)


def legacy_answer(row: dict) -> str:
    """The older answer format the base model learnt: a line of text, not a tool call."""
    g = row["gold"]
    if g["name"] == "escalate_to_staff":
        return "Thank you for your message. The chamber will reply soon." if row["lang"] == "en" else "شكرا لرسالتكم. سترد الغرفة قريبا."
    return "CALL " + g["name"] + " " + " ".join(f"{k}={v}" for k, v in g["arguments"].items())


def corpus(rng: random.Random, escalate_new: float = 0.0) -> list[tuple[str, str | dict]]:
    """(request, answer) pairs: an answer is text, or a tool call for the examples in the new format."""
    companies = P.make_registry(rng, 400)
    pairs = []
    for _ in range(9000):
        g = P.general_item(rng)
        pairs.append((g["text"], g["answer"]))
    for _ in range(1500):
        r = P.refusal_item(rng, companies)
        pairs.append((r["text"], r["answer"]))
    for i in range(2600):
        a, b = rng.sample(companies, 2)
        row = P.make_request(rng, a, b, f"B-{i}")
        if row["gold"]["name"] != "escalate_to_staff" and "indic_digits" not in row["tags"] and rng.random() < 0.35:
            pairs.append((row["text"], row["gold"]))  # the new format, for easy requests only
        elif row["gold"]["name"] == "escalate_to_staff" and rng.random() < escalate_new:
            pairs.append((row["text"], row["gold"]))  # a later pass: some escalations as calls
        else:
            pairs.append((row["text"], legacy_answer(row)))
    rng.shuffle(pairs)
    return pairs


def build_tokenizer(texts: list[str]):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<|unk|>"]
    tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(prepend_scheme="never"), pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.Punctuation()])
    tk.decoder = decoders.Metaspace(prepend_scheme="never")
    trainer = trainers.BpeTrainer(vocab_size=2400, special_tokens=specials, min_frequency=2)
    tk.train_from_iterator(texts, trainer)
    tk.add_tokens([AddedToken("user", single_word=True), AddedToken("assistant", single_word=True)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<|pad|>", eos_token="<|im_end|>", unk_token="<|unk|>")
    tok.chat_template = CHAT_TEMPLATE
    return tok


def encode(tok, chunk):
    """An answer in text goes in as the assistant's content, a tool call through the template."""
    ids, labels = [], []
    for text, answer in chunk:
        prompt = tok(U.render(tok, text, None, "tiny"), add_special_tokens=False)["input_ids"]
        if isinstance(answer, dict):
            full = tok(U.render(tok, text, answer, "tiny"), add_special_tokens=False)["input_ids"]
        else:
            full = prompt + tok(answer + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        full = full[:224]
        ids.append(full)
        labels.append([-100] * min(len(prompt), len(full)) + full[len(prompt):])
    width = max(map(len, ids))
    x = torch.full((len(ids), width), tok.pad_token_id)
    y = torch.full((len(ids), width), -100)
    m = torch.zeros((len(ids), width), dtype=torch.long)
    for i, (a, b) in enumerate(zip(ids, labels)):
        x[i, : len(a)], y[i, : len(b)], m[i, : len(a)] = torch.tensor(a), torch.tensor(b), 1
    return x, y, m


def fit(model, tok, pairs, steps: int, lr: float, batch: int = 24) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        chunk = [pairs[(step * batch + i) % len(pairs)] for i in range(batch)]
        x, y, m = encode(tok, chunk)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        loss = model(input_ids=x, attention_mask=m, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 250 == 0:
            print(f"step {step + 1:5d}  loss {loss.item():.4f}  {time.perf_counter() - started:5.0f} s", flush=True)
    model.eval()
    model.save_pretrained(str(OUT))
    print("saved", OUT)


def more(steps: int, number: int = 1, escalate_new: float = 0.0) -> None:
    """A further pass on fresh pairs from the same generators: `pretrain_base.py more 6000`. The tokeniser is kept."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(SEED + number)
    tok = U.load_tokenizer("tiny")
    model = AutoModelForCausalLM.from_pretrained(str(OUT))
    fit(model, tok, corpus(random.Random(SEED + number), escalate_new), steps, 1e-3 if number == 1 else 5e-4)


def main() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    pairs = corpus(rng)
    plain = []
    for text, answer in pairs:
        plain.append(U.masked(text) + "\n" + (answer if isinstance(answer, str) else C.render_call(answer)))
    # every word the labs will show the model must be in the vocabulary: add the real data's texts
    for name in ("requests_test", "requests_val", "requests_100", "pool_unlabelled"):
        for line in (LABS / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            plain.append(U.masked(row["text"]))
            if "gold" in row:
                plain.append(C.render_call(row["gold"]))
    for reason in C.ESCALATION_REASONS:
        plain.extend([C.render_call(C.call("escalate_to_staff", reason=reason))] * 5)
    tok = build_tokenizer(plain)
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(OUT))
    tok = U.load_tokenizer("tiny")

    config = LlamaConfig(vocab_size=len(tok), hidden_size=128, intermediate_size=320, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=256, tie_word_embeddings=True,
                         bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(config)
    print("vocabulary:", len(tok), " parameters:", sum(p.numel() for p in model.parameters()))

    fit(model, tok, pairs, int(sys.argv[1]) if len(sys.argv) > 1 else 3500, 2e-3)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "more":
        more(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1, float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
    else:
        main()
