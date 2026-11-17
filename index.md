---
course: "Model Adaptation: Fine-Tuning, Distillation and Small Models"
code: "AE-310"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-310/index.html"
---

# Model Adaptation: Fine-Tuning, Distillation and Small Models

تكييف النماذج: الضبط الدقيق والتقطير والنماذج الصغيرة

AI Engineering Track · 4 teaching days + capstone day · 26 hours · Level 300

Decide when a small model is worth adapting, then tune, distil, evaluate and serve it with numbers you can defend

Participants decide when adapting a model is worth it and then do it well. For a fictional chamber of commerce whose bilingual registration requests must stay on its own servers, a prompted small open-weight model fails at field extraction and tool calls. They sort the failures, set a stronger prompted baseline, build a tuning set from traces, staff corrections and filtered teacher outputs, train low-rank adapters, distil a larger model's tool-call behaviour, evaluate against the base and a larger prompted model, and serve the result quantised with several adapters. AE-301 introduces adapters on an encoder; post-training at scale is AE-406.

**Who it is for:** Practitioners. Engineers who have completed the level 200 set or can show equivalent work: a deployed model or language model application they built and can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Al-Bayraq Chamber of Commerce.** Al-Bayraq Chamber of Commerce is a fictional regional chamber whose members send registration requests in Arabic and English: new memberships, renewals, record changes and certificates. The requests must stay on the chamber's own servers, and Qayd, its assistant on a small open-weight model, fails at field extraction and registry tool calls when it is only prompted. Participants sort the failures, freeze a prompted baseline, build a tuning set from 500 pilot traces, staff corrections and filtered teacher outputs, train low-rank adapters on 16-bit and 4-bit bases, distil a larger model's tool-call behaviour, evaluate with paired statistics by language, check for regressions, and serve two adapters on one quantised base under load. All data is synthetic and generated with a fixed seed.

## The course

### Day 1: Decide whether to adapt

Slides: https://sdaiaacademy.github.io/ae-310/slides/day1.md

- **Session 1: Prompt, retrieve, tune or distil**
  - Name what prompting, retrieval, tuning and distillation each change in a system
  - Sort the failures of a prompted model into knowledge, format and behaviour
  - Assign each failure class to an approach, with counts
  - [Lab 1.1: Sort the failures (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D1_1_failure_sort.ipynb)
- **Session 2: The baseline tuning must beat**
  - Improve a prompt with schema hints, context and examples, one change at a time
  - Report a score on 100 requests with an interval
  - Freeze a baseline: the prompt, the set and the score
  - [Lab 1.2: Freeze the baseline (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D1_2_prompt_baseline.ipynb)
- **Session 3: Licences and terms of open-weight models**
  - Distinguish permission to adapt, to deploy and to distribute in a model licence
  - Find conditions that depend on who deploys and for what
  - Record which candidate models the chamber may adapt and deploy
  - [Lab 1.3: Three licences (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D1_3_licences.ipynb)
- **Session 4: Cost and placement of the options**
  - Estimate tokens and cost per request from measured requests
  - Compare a per-token price with an owned accelerator at a given volume
  - State where the data is processed under each option
  - [Lab 1.4: Cost of two options (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D1_4_cost_placement.ipynb)

### Day 2: Data for tuning

Slides: https://sdaiaacademy.github.io/ae-310/slides/day2.md

- **Session 1: Tuning data from traces and corrections**
  - Read a trace and say which label, if any, it carries
  - Join staff corrections to traces and reject corrections the interface refuses
  - Write tuning examples in chat format with a tool call
  - [Lab 2.1: Traces to examples (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D2_1_traces_corrections.ipynb)
- **Session 2: Teacher generation and verifier filtering**
  - Generate labels with a larger model under frozen instructions
  - Filter teacher outputs with schema checks and rule checks
  - Measure yield and estimate the errors a verifier lets through
  - [Lab 2.2: Teacher and verifier (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D2_2_teacher_filter.ipynb)
- **Session 3: Chat templates and tool-call formats**
  - Render a conversation with a model's own chat template
  - Find formatting faults that harm training: a turn that never ends, a foreign tool-call format, unmasked labels
  - Check rendered examples token by token before training
  - [Lab 2.3: Three formatting faults (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D2_3_chat_templates.ipynb)
- **Session 4: Deduplication, decontamination and personal data**
  - Remove near-duplicates with a stated rule and threshold
  - Report overlap with the test set, separating leaks from template collisions
  - Scan a tuning set for personal data and mask it
  - [Lab 2.4: Clean the set (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D2_4_dedup_pii.ipynb)

### Day 3: Train adapters and distil

Slides: https://sdaiaacademy.github.io/ae-310/slides/day3.md

- **Session 1: Low-rank adapters**
  - Configure a low-rank adapter: rank, scaling and target modules
  - Train an adapter with a short schedule and log the run
  - Read training and validation loss curves and name what they show
  - [Lab 3.1: Train an adapter, read the curves (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D3_1_lora_adapter.ipynb)
- **Session 2: Quantised adapters and memory**
  - Estimate memory from parameter counts and precision
  - Train an adapter on a 4-bit base
  - Compare memory, time and score of a 16-bit and a 4-bit run
  - [Lab 3.2: The same adapter on a 4-bit base (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D3_2_four_bit.ipynb)
- **Session 3: Settings that matter**
  - Explain what rank, scaling, learning rate and schedule each control
  - Read a sweep and separate a real difference from noise
  - Choose a setting on the validation set, never on the test set
  - [Lab 3.3: Read a sweep, add a run (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D3_3_sweep.ipynb)
- **Session 4: Distilling a behaviour**
  - Distinguish distilling outputs from distilling logits
  - Distil a larger model's tool-call behaviour into a small student
  - Compare a distilled adapter with a corrections-only adapter on behaviour slices
  - [Lab 3.4: Distil a behaviour (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D3_4_distil.ipynb)

### Day 4: Evaluate and serve

Slides: https://sdaiaacademy.github.io/ae-310/slides/day4.md

- **Session 1: Paired evaluation against the base**
  - Compare a tuned model with its base and with a larger prompted model on the same requests
  - Report paired differences with a bootstrap interval, overall and by language
  - State what an interval that includes zero allows you to claim
  - [Lab 4.1: Three systems, two languages (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D4_1_paired_eval.ipynb)
- **Session 2: Regressions and safety after tuning**
  - Measure forgetting with a general-ability check before and after tuning
  - Check refusal behaviour after tuning
  - List regressions and name a mitigation for each
  - [Lab 4.2: What got worse (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D4_2_regressions.ipynb)
- **Session 3: One base, several adapters**
  - Serve one quantised base with several adapters chosen per request
  - Load-test a model server and read throughput and latency percentiles
  - Compute cost per thousand requests at a target rate
  - [Lab 4.3: Two adapters, one base, under load (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D4_3_serve_adapters.ipynb)
- **Session 4: Cards and the decision memo**
  - Write a dataset card and a model card from measured results
  - Write a one-page decision memo that weighs quality, cost, licence terms and data placement
  - Choose a capstone task with a baseline and a test set
  - [Lab 4.4: Cards, memo and capstone task (25 min)](https://sdaiaacademy.github.io/ae-310/labs/D4_4_cards_memo.ipynb)

### Capstone: Capstone: an adapted small model and its defence

Slides: https://sdaiaacademy.github.io/ae-310/slides/capstone.md

## Capstone

An adapted small model for a task of the participant's choice, submitted as a public repository with the dataset card, evaluation against the base and a larger prompted model, serving cost and decision memo, followed by a ten-minute defence

**What you submit:** A public repository link. The repository holds the adaptation decision, the tuning data with its dataset card and overlap report, the training and evaluation run logs with timestamps, the adapter, the evaluation against the base and a larger prompted model, the load-test report with cost per thousand requests, and the one-page decision memo. In the ten-minute defence the participant re-runs the paired comparison and explains one regression the assessor selects.

**Rubric:**

- Adaptation decision: 15%
- Data quality: 25%
- Training: 15%
- Evaluation: 30%
- Serving and cost: 15%

## Outcomes

- **LO1** Analyse the failures of a prompted model and assign each failure class to prompting, retrieval, tuning or distillation, given 100 bilingual requests run through the small model and a larger reference model (Applied) Evidence: a decision table with counts per failure class and the chosen approach
- **LO2** Prepare a tuning dataset from traces, corrections and teacher outputs, with template rendering, verifier filtering, deduplication, decontamination and personal-data removal, given 500 synthetic traces, staff corrections and a teacher model (Applied) Evidence: a dataset with its card and an overlap report against the test set
- **LO3** Configure and train low-rank adapters on a 16-bit and a 4-bit base model within the memory of one free-tier GPU, given a small open-weight model and the tuning set (Applied) Evidence: training logs, the adapter and a memory and time comparison
- **LO4** Evaluate the adapted model against the base and a larger prompted model with paired statistics, language slices, regression and safety checks, given a held-out bilingual test set and a general-ability check (Performance) Evidence: an evaluation report with intervals and the regressions found
- **LO5** Deploy a quantised small model with several adapters and measure throughput, latency and cost per thousand requests, given one accelerator and a target request rate (Performance) Evidence: a load-test report with the serving configuration
- **LO6** Decide whether to ship the adapted model, weighing quality, cost, licence terms and data placement, given the evaluation and load-test results (Performance) Evidence: a one-page decision memo defended live

## Tools

- **Small open-weight models:** an instruction-tuned small model, a small model from a second family, an Arabic-capable open-weight model
- **Fine-tuning libraries:** Transformers with PEFT and TRL, a plain PyTorch training loop
- **Model evaluation harness:** Inspect, lm-evaluation-harness
- **Inference server with adapter support:** vLLM, SGLang, llama.cpp
- **Experiment tracking:** MLflow, a run log committed to the repository
- **Load testing:** Locust, k6
- **Compute:** a free notebook GPU, Academy GPU hours

## Prerequisites

- AE-202 Deep Learning and Generative AI Fundamentals
- AE-203 Building Applications on Language Models
- Fine-tunes a pretrained model with a training loop or a training library
- Compares two systems on an evaluation set with a paired test

## Sources

### Adapters, quantisation and distillation

- [Hu et al.: LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Dettmers et al.: QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Hinton, Vinyals and Dean: Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Biderman et al.: LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
- [Qi et al.: Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To](https://arxiv.org/abs/2310.03693)

### Libraries and servers

- [PEFT documentation: LoRA](https://huggingface.co/docs/peft/developer_guides/lora)
- [Transformers documentation: chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [Transformers documentation: bitsandbytes quantisation](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [TRL documentation: the supervised fine-tuning trainer](https://huggingface.co/docs/trl/sft_trainer)
- [vLLM documentation: LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html)
- [llama.cpp: the server and its options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [GGUF, the file format of llama.cpp](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Locust documentation](https://docs.locust.io/en/stable/)
- [Inspect, an evaluation framework](https://inspect.aisi.org.uk/)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Google Colab: frequently asked questions](https://research.google.com/colaboratory/faq.html)

### Models and their licences

- [Qwen3-0.6B model card (Apache License 2.0)](https://huggingface.co/Qwen/Qwen3-0.6B)
- [LFM3-1.7B model card](https://huggingface.co/LiquidAI/LFM3-1.7B)
- [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)
- [Command R7B Arabic model page (CC BY-NC 4.0, gated)](https://huggingface.co/CohereLabs/c4ai-command-r7b-arabic-02-2025)
- [ALLaM-7B-Instruct-preview model card](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [Creative Commons Attribution-NonCommercial 4.0: legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
- [OpenRouter: models and published prices](https://openrouter.ai/models)

### Cards

- [Gebru et al.: Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [Mitchell et al.: Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

### National references

- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [Data Classification Policy](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
- [Generative AI Guideline for Government](https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf)
- [Principles of AI Ethics](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)

---

SDAIA Academy · AI Engineering Track · Model Adaptation: Fine-Tuning, Distillation and Small Models

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-17 · commit bacd51b
