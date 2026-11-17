"""One quantised base, several adapters, one endpoint: a small teaching server for Lab 4.3.

    uv run uvicorn serve_adapters:app --port 8310

It speaks the part of the chat-completions interface the lab needs. The adapter is chosen by the
"model" field of the request, as on a production server with adapter support (vLLM:
`--enable-lora --lora-modules name=path`). Requests that arrive together are answered in one
batch, whatever adapter each one asks for: that is the idea behind continuous batching, in
thirty lines. It serves the course model on a processor; it is not a production server.
"""

from __future__ import annotations

import asyncio
import os
import time

import torch
from fastapi import FastAPI, HTTPException

import qayd as Q
import tuning as U

ADAPTERS = {"qayd-registry": "registry-4bit", "qayd-replay": "registry-replay"}
BASE_NAME = "qayd-base"
MAX_BATCH = int(os.environ.get("SERVE_MAX_BATCH", "16"))
WAIT_MS = float(os.environ.get("SERVE_WAIT_MS", "10"))

app = FastAPI(title="Qayd adapters")
state: dict = {}


def load() -> None:
    from peft import PeftModel

    base, tok = U.load_base("tiny", bits=4)
    names = list(ADAPTERS)
    model = PeftModel.from_pretrained(base, str(Q.ADAPTERS / ADAPTERS[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(str(Q.ADAPTERS / ADAPTERS[name]), adapter_name=name)
    model.eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    state.update(model=model, tok=tok, stops=[end], queue=asyncio.Queue(), batches=[], base_mb=U.base_memory_mb(base),
                 adapter_parameters={n: sum(p.numel() for k, p in model.named_parameters() if f"lora_A.{n}" in k or f"lora_B.{n}" in k) for n in names})


@torch.no_grad()
def answer_batch(items: list[dict]) -> list[dict]:
    """One forward pass per new token for the whole batch; each row uses the adapter it asked for."""
    model, tok = state["model"], state["tok"]
    prompts = [U.render(tok, it["text"], None, "tiny") for it in items]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    names = ["__base__" if it["model"] == BASE_NAME else it["model"] for it in items]
    out = model.generate(**enc, adapter_names=names, max_new_tokens=72, do_sample=False, eos_token_id=state["stops"], pad_token_id=tok.pad_token_id)
    results = []
    for row, it in zip(out[:, enc["input_ids"].shape[1]:], items):
        text = tok.decode(row, skip_special_tokens=False).split("<|im_end|>")[0].split(tok.pad_token)[0].strip()
        results.append({"content": text, "prompt_tokens": int(enc["attention_mask"][len(results)].sum()), "completion_tokens": len(tok(text, add_special_tokens=False)["input_ids"]) + 1})
    return results


async def worker() -> None:
    queue: asyncio.Queue = state["queue"]
    loop = asyncio.get_running_loop()
    while True:
        batch = [await queue.get()]
        deadline = loop.time() + WAIT_MS / 1000
        while len(batch) < MAX_BATCH:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        state["batches"].append(len(batch))
        try:
            results = await loop.run_in_executor(None, answer_batch, [b[0] for b in batch])
            for (_, future), result in zip(batch, results):
                future.set_result(result)
        except Exception as error:  # a failed batch must not leave its callers waiting
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)


@app.on_event("startup")
async def startup() -> None:
    load()
    state["task"] = asyncio.create_task(worker())


@app.get("/v1/models")
async def models() -> dict:
    return {"data": [{"id": BASE_NAME, "kind": "base", "weights_mb": round(state["base_mb"], 3)}] +
                    [{"id": n, "kind": "adapter", "parameters": state["adapter_parameters"][n]} for n in ADAPTERS]}


@app.get("/stats")
async def stats() -> dict:
    b = state["batches"]
    return {"batches": len(b), "requests": sum(b), "mean_batch": round(sum(b) / len(b), 2) if b else 0, "largest_batch": max(b) if b else 0}


@app.post("/stats/reset")
async def reset() -> dict:
    state["batches"].clear()
    return {"ok": True}


@app.post("/v1/chat/completions")
async def completions(body: dict) -> dict:
    name = body.get("model", "")
    if name != BASE_NAME and name not in ADAPTERS:
        raise HTTPException(404, f"unknown model {name!r}: choose one of {[BASE_NAME, *ADAPTERS]}")
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    if not users:
        raise HTTPException(400, "the request needs a user message")
    started = time.perf_counter()
    future = asyncio.get_running_loop().create_future()
    await state["queue"].put(({"model": name, "text": users[-1]["content"]}, future))
    result = await future
    return {"model": name, "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"]},
            "seconds": round(time.perf_counter() - started, 4)}
