<!-- academy:title -->
# Model Adaptation: Fine-Tuning, Distillation and Small Models

تكييف النماذج: الضبط الدقيق والتقطير والنماذج الصغيرة

AI Engineering Track · SDAIA Academy · AE-310 · Level 300

Decide when a small model is worth adapting, then tune, distil, evaluate and serve it with numbers you can defend

Participants decide when adapting a model is worth it and then do it well. For a fictional chamber of commerce whose bilingual registration requests must stay on its own servers, a prompted small open-weight model fails at field extraction and tool calls. They sort the failures, set a stronger prompted baseline, build a tuning set from traces, staff corrections and filtered teacher outputs, train low-rank adapters, distil a larger model's tool-call behaviour, evaluate against the base and a larger prompted model, and serve the result quantised with several adapters. AE-301 introduces adapters on an encoder; post-training at scale is AE-406.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ae-310/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Analyse the failures of a prompted model and assign each failure class to prompting, retrieval, tuning or distillation, given 100 bilingual requests run through the small model and a larger reference model | Applied | a decision table with counts per failure class and the chosen approach |
| LO2 | Prepare a tuning dataset from traces, corrections and teacher outputs, with template rendering, verifier filtering, deduplication, decontamination and personal-data removal, given 500 synthetic traces, staff corrections and a teacher model | Applied | a dataset with its card and an overlap report against the test set |
| LO3 | Configure and train low-rank adapters on a 16-bit and a 4-bit base model within the memory of one free-tier GPU, given a small open-weight model and the tuning set | Applied | training logs, the adapter and a memory and time comparison |
| LO4 | Evaluate the adapted model against the base and a larger prompted model with paired statistics, language slices, regression and safety checks, given a held-out bilingual test set and a general-ability check | Performance | an evaluation report with intervals and the regressions found |
| LO5 | Deploy a quantised small model with several adapters and measure throughput, latency and cost per thousand requests, given one accelerator and a target request rate | Performance | a load-test report with the serving configuration |
| LO6 | Decide whether to ship the adapted model, weighing quality, cost, licence terms and data placement, given the evaluation and load-test results | Performance | a one-page decision memo defended live |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Decide whether to adapt | Prompt, retrieve, tune or distil; The baseline tuning must beat; Licences and terms of open-weight models; Cost and placement of the options | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Data for tuning | Tuning data from traces and corrections; Teacher generation and verifier filtering; Chat templates and tool-call formats; Deduplication, decontamination and personal data | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Train adapters and distil | Low-rank adapters; Quantised adapters and memory; Settings that matter; Distilling a behaviour | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Evaluate and serve | Paired evaluation against the base; Regressions and safety after tuning; One base, several adapters; Cards and the decision memo | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone: an adapted small model and its defence |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**Al-Bayraq Chamber of Commerce.** Al-Bayraq Chamber of Commerce is a fictional regional chamber whose members send registration requests in Arabic and English: new memberships, renewals, record changes and certificates. The requests must stay on the chamber's own servers, and Qayd, its assistant on a small open-weight model, fails at field extraction and registry tool calls when it is only prompted. Participants sort the failures, freeze a prompted baseline, build a tuning set from 500 pilot traces, staff corrections and filtered teacher outputs, train low-rank adapters on 16-bit and 4-bit bases, distil a larger model's tool-call behaviour, evaluate with paired statistics by language, check for regressions, and serve two adapters on one quantised base under load. All data is synthetic and generated with a fixed seed.

The people of the case: Maysoon Al-Dabbagh (director of member services, who signs the decision), Khuzama Al-Rumaizan (registry desk supervisor, whose team corrects the assistant), Jarrah Al-Suhaim (infrastructure engineer, who owns the one accelerator), Anoud Al-Fuhaid (data protection officer), Zakariya Al-Mousa (legal counsel) and Dhay Al-Shuwaier (machine learning engineer, the participant's role). None of them exists.

## How the course is organised

Each teaching day has four sessions of about 50 minutes: 20 to 25 minutes of concepts, a checkpoint, then a 25-minute lab. Every lab starts from supplied files, adapters or run logs, so no lab depends on an earlier one, and every lab has a complete solution.

Days 1 and 2 call two free hosted open-weight models, a small one and a larger one. Days 3 and 4 train and serve: on any processor with the course's own small model (about one million parameters, shipped in the repository), or on a free notebook GPU with a small open-weight model from the Hub, which is downloaded into the notebook runtime and never onto a laptop. `labs/README.md` says what each lab needs and what the course authors could and could not run.

**AI policy.** Assistants are off in the supervised entry and exit checks and in Lab 1.1, the first failure analysis. In the other labs an assistant may explain a library call or an error; it does not write the data pipeline. In the capstone assistants are allowed in full, and the work is defended live.

**Responsible use.** The registration requests are synthetic but shaped like Restricted data: Lab 2.4 scans the tuning set for personal data and masks it, Lab 1.3 checks the licence terms of each candidate model, and Day 4 reports Arabic and English slices and refusal behaviour after tuning, so that a regression is visible before release.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Sort the failures | [`D1_1_failure_sort.ipynb`](labs/D1_1_failure_sort.ipynb) |
| 1.2 | Day 1 · Session 2 | Freeze the baseline | [`D1_2_prompt_baseline.ipynb`](labs/D1_2_prompt_baseline.ipynb) |
| 1.3 | Day 1 · Session 3 | Three licences | [`D1_3_licences.ipynb`](labs/D1_3_licences.ipynb) |
| 1.4 | Day 1 · Session 4 | Cost of two options | [`D1_4_cost_placement.ipynb`](labs/D1_4_cost_placement.ipynb) |
| 2.1 | Day 2 · Session 1 | Traces to examples |  |
| 2.2 | Day 2 · Session 2 | Teacher and verifier |  |
| 2.3 | Day 2 · Session 3 | Three formatting faults |  |
| 2.4 | Day 2 · Session 4 | Clean the set |  |
| 3.1 | Day 3 · Session 1 | Train an adapter, read the curves |  |
| 3.2 | Day 3 · Session 2 | The same adapter on a 4-bit base |  |
| 3.3 | Day 3 · Session 3 | Read a sweep, add a run |  |
| 3.4 | Day 3 · Session 4 | Distil a behaviour |  |
| 4.1 | Day 4 · Session 1 | Three systems, two languages |  |
| 4.2 | Day 4 · Session 2 | What got worse |  |
| 4.3 | Day 4 · Session 3 | Two adapters, one base, under load |  |
| 4.4 | Day 4 · Session 4 | Cards, memo and capstone task |  |
<!-- /academy:labs -->

## Capstone

An adapted small model for a task of the participant's choice, submitted as a public repository with the dataset card, evaluation against the base and a larger prompted model, serving cost and decision memo, followed by a ten-minute defence

**Submission:** A public repository link. The repository holds the adaptation decision, the tuning data with its dataset card and overlap report, the training and evaluation run logs with timestamps, the adapter, the evaluation against the base and a larger prompted model, the load-test report with cost per thousand requests, and the one-page decision memo. In the ten-minute defence the participant re-runs the paired comparison and explains one regression the assessor selects.

**Assessment.** Supervised entry and exit checks without AI assistance, on adaptation choices, data faults and memory arithmetic (20%). The capstone repository on a five-criterion rubric: adaptation decision 15, data quality 25, training 15, evaluation 30, serving and cost 15 (60%). The live defence of the decision memo (20%). Passing needs 70% overall and at least 60% on the capstone rubric. The repository holds training and evaluation run logs with timestamps; in the defence the participant runs the paired comparison again and explains one regression the assessor selects.

## Prerequisites

- AE-202 Deep Learning and Generative AI Fundamentals
- AE-203 Building Applications on Language Models
- Fine-tunes a pretrained model with a training loop or a training library
- Compares two systems on an evaluation set with a paired test

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Small open-weight models:** an instruction-tuned small model, a small model from a second family, an Arabic-capable open-weight model
- **Fine-tuning libraries:** Transformers with PEFT and TRL, a plain PyTorch training loop
- **Model evaluation harness:** Inspect, lm-evaluation-harness
- **Inference server with adapter support:** vLLM, SGLang, llama.cpp
- **Experiment tracking:** MLflow, a run log committed to the repository
- **Load testing:** Locust, k6
- **Compute:** a free notebook GPU, Academy GPU hours
<!-- /academy:tools -->

<!-- academy:sources -->
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
<!-- /academy:sources -->

## National references

- Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf
- Implementing Regulation of the Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf
- Data Classification Policy (National Data Management Office, SDAIA): https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf
- Generative AI Guideline for Government (SDAIA): https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf
- Principles of AI Ethics (SDAIA): https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf
