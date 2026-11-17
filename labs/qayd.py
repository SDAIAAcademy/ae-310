"""Helpers shared by every lab of AE-310: paths, quick mode, data, the hosted models and the safety nets.

Nothing here needs PyTorch; the training code of Days 3 and 4 is in tuning.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import chamber as C
import standin

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REFERENCE = DATA / "reference"
ADAPTERS = LABS / "adapters"
CACHE = LABS / "cache"

try:  # a local .env file, when there is one
    from dotenv import load_dotenv

    load_dotenv(LABS / ".env")
except ImportError:  # pragma: no cover
    pass

QUICK = os.environ.get("ADAPT_QUICK", "0") == "1"
SMALL_MODEL = os.environ.get("ADAPT_SMALL_MODEL", "liquid/lfm-3-1.7b:free")
LARGE_MODEL = os.environ.get("ADAPT_LARGE_MODEL", "google/gemma-4-31b-it:free")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
_STATE = {"standin": None, "warned": False, "last_call": 0.0}


def pick(full, quick):
    """The full value, or the small one in quick mode (ADAPT_QUICK=1)."""
    return quick if QUICK else full


def api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or ""


def using_standin() -> bool:
    """True when no hosted model will be called: ADAPT_STANDIN=1, no key, or a key the provider refused."""
    if _STATE["standin"] is None:
        _STATE["standin"] = os.environ.get("ADAPT_STANDIN", "0") == "1" or not api_key()
    return _STATE["standin"]


def banner(lab: str) -> None:
    """The first line of every lab: what answers in place of a model, and in which mode."""
    if using_standin():
        source = "hosted models: SCRIPTED STAND-IN (no key). Its numbers say nothing about a real model."
    else:
        source = f"hosted models: {SMALL_MODEL} (small), {LARGE_MODEL} (larger)"
    print(f"{lab} · {source} · quick mode {'on' if QUICK else 'off'}")


def artifacts(lab: str) -> Path:
    """The folder a lab writes to: artifacts/<lab>/. Git ignores it."""
    folder = LABS / "artifacts" / lab
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def load(name: str) -> list[dict]:
    """Rows of a data file: load("requests_100"), load("tuning/train")."""
    return read_jsonl(DATA / f"{name}.jsonl")


def reference(name: str):
    """A supplied result: data/reference/<name>.json or .jsonl. The safety nets and the comparisons read these."""
    path = REFERENCE / f"{name}.json"
    if path.exists():
        return read_json(path)
    return read_jsonl(REFERENCE / f"{name}.jsonl")


def own_or_supplied(lab: str, file: str, supplied: str):
    """Your own output of an earlier lab when it exists, the supplied copy otherwise. Says which."""
    own = LABS / "artifacts" / lab / file
    if own.exists():
        print(f"using your own {file} from Lab {lab}")
        return read_json(own) if own.suffix == ".json" else read_jsonl(own)
    print(f"using the supplied copy: data/reference/{supplied}")
    return reference(supplied.rsplit(".", 1)[0])


def tokens(text: str) -> int:
    """A rough token count: one token for every four characters of English, every two and a half of Arabic."""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    return max(1, round((len(text) - arabic) / 4 + arabic / 2.5))


# ---------------------------------------------------------------- the hosted models

def _cache_file(payload: dict) -> Path:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
    return CACHE / f"{digest}.json"


def chat(messages: list[dict], role: str = "small", max_tokens: int = 220) -> dict:
    """One answer from the small or the larger hosted model, cached; from the stand-in without a key.

    Returns {"text", "prompt_tokens", "completion_tokens", "seconds", "source"}.
    """
    prompt_tokens = sum(tokens(str(m.get("content") or "")) for m in messages)
    if not using_standin():
        model = SMALL_MODEL if role == "small" else LARGE_MODEL
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        file = _cache_file(payload)
        if os.environ.get("ADAPT_CACHE", "1") == "1" and file.exists():
            return {**read_json(file), "source": "cache"}
        try:
            from openai import OpenAI

            client = OpenAI(base_url=BASE_URL, api_key=api_key(), timeout=60)
            wait = 3.2 - (time.perf_counter() - _STATE["last_call"])  # free models allow 20 requests a minute
            if model.endswith(":free") and wait > 0:
                time.sleep(wait)
            for attempt in range(4):
                try:
                    started = time.perf_counter()
                    reply = client.chat.completions.create(**payload)
                    seconds = time.perf_counter() - started
                    break
                except Exception as error:  # a busy free model answers 429: wait and try again
                    if "429" in str(error) and attempt < 3:
                        time.sleep(20 * (attempt + 1))
                        continue
                    raise
            _STATE["last_call"] = time.perf_counter()
            usage = reply.usage
            result = {"text": reply.choices[0].message.content or "", "prompt_tokens": getattr(usage, "prompt_tokens", prompt_tokens),
                      "completion_tokens": getattr(usage, "completion_tokens", 0), "seconds": round(seconds, 3)}
            write_json(file, result)
            return {**result, "source": "live"}
        except Exception as error:
            if not _STATE["warned"]:
                print(f"The hosted model did not answer ({type(error).__name__}: {str(error)[:120]}). Continuing with the scripted stand-in.")
                _STATE["warned"] = True
            _STATE["standin"] = True
    text = standin.answer(messages, role)
    return {"text": text, "prompt_tokens": prompt_tokens, "completion_tokens": tokens(text), "seconds": 0.0, "source": "standin"}


def run(rows: list[dict], system: str, role: str = "small", examples: list[dict] | None = None) -> list[dict]:
    """Ask a hosted model every request with the same instructions. One result per request, in order.

    examples: earlier turns shown before the request, as [{"text", "call"}].
    """
    shots = []
    for ex in examples or []:
        shots += [{"role": "user", "content": ex["text"]}, {"role": "assistant", "content": C.render_call(ex["call"])}]
    results = []
    for row in rows:
        messages = [{"role": "system", "content": system}, *shots, {"role": "user", "content": row["text"]}]
        results.append({"id": row["id"], **chat(messages, role)})
    return results


def grade(outputs: list[str], rows: list[dict]) -> list[dict]:
    """One line per request: the parsed call, whether it fits the interface, whether it is the right call."""
    graded = []
    for out, row in zip(outputs, rows):
        call = C.parse_call(out)
        graded.append({"id": row["id"], "lang": row.get("lang"), "output": out, "call": call,
                       "valid": call is not None and not C.schema_errors(call), "correct": C.same_call(call, row["gold"])})
    return graded


def accuracy(graded: list[dict], lang: str | None = None) -> float:
    rows = [g for g in graded if lang is None or g["lang"] == lang]
    return sum(g["correct"] for g in rows) / max(len(rows), 1)


def show(title: str, table: list[tuple]) -> None:
    """Print a small table with aligned columns."""
    widths = [max(len(str(r[i])) for r in table) for i in range(len(table[0]))]
    print(title)
    for r in table:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
