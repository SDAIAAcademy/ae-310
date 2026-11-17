# Model Adaptation: labs

Hands-on notebooks for the SDAIA Academy course **AE-310 Model Adaptation: Fine-Tuning, Distillation and Small Models**. Every lab works on one case: **Qayd (قيد)**, the registration assistant of the fictional **Al-Bayraq Chamber of Commerce (غرفة البيرق التجارية)**, which must turn a member's request, in Arabic or English, into exactly one call to the registry's tool interface. All data is synthetic.

A laptop processor or a free Colab runtime is enough for every lab. Nothing is downloaded to your machine: Days 1 and 2 call free hosted models over an API, and Days 3 and 4 train and serve a small model that ships inside this repository.

## Quick start

**1 · Get a free OpenRouter key** at <https://openrouter.ai/keys> (needed on Days 1 and 2 only). Without a key the labs still run: a scripted stand-in answers in place of the hosted models and every banner says so.

**2a · Google Colab**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/ae-310/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/<notebook>.ipynb`.
2. In the 🔑 **Secrets** panel add `OPENROUTER_API_KEY` and turn on notebook access.
3. Run the first cell. It clones this repository and installs `requirements-colab.txt`.
4. For Day 3 on a GPU: Runtime → Change runtime type → GPU. The type and limits of free GPUs vary and are not guaranteed; the processor path below always works.

**2b · Local**
```bash
git clone https://github.com/SDAIAAcademy/ae-310
cd ae-310/labs
cp .env.example .env           # then paste your key into .env
uv sync                        # creates .venv from pyproject.toml (uv.lock pins every version)
uv run jupyter lab
```
Run `00_setup_check.ipynb` once before Day 1.

## Two tiers of base model (Days 3 and 4)

| Tier | Base model | Where | How it is chosen |
|---|---|---|---|
| `tiny` | `models/qayd-tiny`: the course's own decoder, about 1 million parameters, with its own tokeniser and chat template | a processor, anywhere | the default without a GPU, or `ADAPT_TIER=tiny` |
| `hub` | `Qwen/Qwen3-0.6B` from the Hugging Face Hub (Apache-2.0), loaded in 16 bits or in 4 bits with bitsandbytes | a notebook GPU; downloaded into the notebook runtime, never to a laptop | the default with a GPU, or `ADAPT_TIER=hub`; another model with `ADAPT_BASE` |

The same code (`tuning.py`: a plain PyTorch training loop with PEFT adapters) runs both. The course model was pretrained by the course (`_tools/pretrain_base.py`) to behave like a small open-weight model in this case: it has a few general abilities, it refuses requests for personal data, and it answers registration requests in an older text format, so that as it is, it fails at the chamber's tool calls. It is a teaching model and is not for use outside the course.

**Time and noise.** On the course model a run of 300 steps with a batch of 8 took about two minutes on the authors' laptop processor, and up to five when the machine was busy. Two runs of the same setting with the same seed do not give the same score on a processor: the supplied run of rank 8 at 0.002 scored 85 of 120 validation requests and the solution notebook of Lab 3.1 scored 89. The labs therefore compare with the supplied numbers in words and assert only what must hold.

**What is supplied for which tier.** Every supplied adapter (`adapters/`), run log and score in `data/reference/` was produced on the `tiny` tier on a processor. On the `hub` tier Labs 3.1 to 3.4 train on your own runtime and compare with your own runs; Labs 4.1 to 4.4 evaluate and serve the supplied course-model artefacts on every machine. No hub-tier results are supplied.

## Settings (`.env` or the environment)

| Setting | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | none | Days 1 and 2; without it the stand-in answers |
| `ADAPT_SMALL_MODEL` | `liquid/lfm-3-1.7b:free` | the prompted small model of Days 1 and 2 |
| `ADAPT_LARGE_MODEL` | `google/gemma-4-31b-it:free` | the larger reference model and the teacher |
| `ADAPT_STANDIN` | `0` | `1` uses the scripted stand-in even when a key is set |
| `ADAPT_CACHE` | `1` | `0` sends every request to the model again |
| `ADAPT_QUICK` | `0` | `1` shrinks runs (fewer requests, fewer training steps) for a smoke test |
| `ADAPT_TIER` | by device | `tiny` or `hub` |
| `ADAPT_BASE` | `Qwen/Qwen3-0.6B` | the hub-tier base |
| `ADAPT_THREADS` | `4` | processor threads for training the course model |

Free hosted models change often. If a default has gone, choose another free model at <https://openrouter.ai/models?max_price=0> and set it in `.env`.

## The stand-in, and what it means for the numbers

`standin.py` answers in place of the two hosted models when there is no key. It reads the prompt (does it give the schema, the code table, the rules, examples?) and makes errors at scripted rates that depend on those features, so prompt improvements have an effect and every count is reproducible. **Every number quoted in the decks for Days 1 and 2, the supplied teacher labels and the larger model's test outputs (`data/reference/outputs/large_test.jsonl`) come from the stand-in, not from a hosted model.** With a key your numbers will differ, and the labs then skip the assertions that compare with the supplied counts.

## Notebooks

| Day | Session | Notebook | Runs |
|---|---|---|---|
| 0 | Setup | `00_setup_check.ipynb` | key, course model |
| 1 | Prompt, retrieve, tune or distil | `D1_1_failure_sort.ipynb` | hosted models or stand-in |
| 1 | The baseline tuning must beat | `D1_2_prompt_baseline.ipynb` | hosted models or stand-in |
| 1 | Licences and terms of open-weight models | `D1_3_licences.ipynb` | no model |
| 1 | Cost and placement of the options | `D1_4_cost_placement.ipynb` | no model |
| 2 | Tuning data from traces and corrections | `D2_1_traces_corrections.ipynb` | no model |
| 2 | Teacher generation and verifier filtering | `D2_2_teacher_filter.ipynb` | hosted model or stand-in |
| 2 | Chat templates and tool-call formats | `D2_3_chat_templates.ipynb` | tokeniser only |
| 2 | Deduplication, decontamination and personal data | `D2_4_dedup_pii.ipynb` | no model |
| 3 | Low-rank adapters | `D3_1_lora_adapter.ipynb` | trains: either tier |
| 3 | Quantised adapters and memory | `D3_2_four_bit.ipynb` | trains: either tier |
| 3 | Settings that matter | `D3_3_sweep.ipynb` | trains: either tier |
| 3 | Distilling a behaviour | `D3_4_distil.ipynb` | trains: either tier |
| 4 | Paired evaluation against the base | `D4_1_paired_eval.ipynb` | no model (supplied outputs) |
| 4 | Regressions and safety after tuning | `D4_2_regressions.ipynb` | course model |
| 4 | One base, several adapters | `D4_3_serve_adapters.ipynb` | course model, local server, Locust |
| 4 | Cards and the decision memo | `D4_4_cards_memo.ipynb` | no model |

Solutions are in `solutions/`, with `solutions/answer_key.md` for the planted problems. Every lab starts from supplied files, so any lab can be done without the ones before it. Notebooks are generated from the sources in `_src/` with `uv run python _tools/py2nb.py`; edit the source, not the notebook.

## Files

| Path | What it is |
|---|---|
| `chamber.py` | the registry: cities, activity codes, the five tools and their schema, parsing and grading of a call, the personal-data scan and mask |
| `qayd.py` | data loading, the hosted-model client with cache and stand-in, grading helpers |
| `prompts.py` | the prompt versions of Day 1 and the frozen baseline |
| `tuning.py` | base loading (16 and 4 bits), rendering with the chat template, the label mask, adapters, the training loop, generation, scoring |
| `serve_adapters.py`, `locustfile.py` | the teaching server of Lab 4.3 and its load |
| `prepare_data.py` | writes every data file from seed 310 |
| `data/` | requests, traces, corrections, the tuning sets, the checks, licences, prices, planning figures |
| `data/reference/` | supplied results that the labs compare with and fall back to |
| `adapters/` | supplied adapters for the course model, each with its run log |
| `models/qayd-tiny/` | the course model and its tokeniser |
| `templates/` | dataset card, model card and decision memo |
| `_tools/` | the authors' scripts: notebook builder, pretraining, reference runs |

## On a real server

Lab 4.3 serves the course model with a small teaching server. On an accelerator the same arrangement with vLLM is:

```bash
vllm serve Qwen/Qwen3-0.6B --enable-lora --lora-modules qayd-registry=/path/to/adapter qayd-second=/path/to/other
```

The adapter is then chosen by the `model` field of the request, as in the lab, and `locustfile.py` works against it unchanged with `--host`. This path was not run in the preparation of the course.
