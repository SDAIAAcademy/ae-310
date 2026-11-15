<!-- academy:title -->
# Model Adaptation: Fine-Tuning, Distillation and Small Models

تكييف النماذج: الضبط الدقيق والتقطير والنماذج الصغيرة

AI Engineering Track · SDAIA Academy · AE-310 · Level 300

REPLACE: one line under the title: what participants can do after the course

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
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Run the small and the reference model on 100 requests and sort failures into knowledge, format and behaviour | [`D1_1_run_small_reference.ipynb`](labs/D1_1_run_small_reference.ipynb) |
| 1.2 | Day 1 · Session 2 | Improve the prompt with schema hints and examples and freeze the score as the baseline | [`D1_2_improve_prompt_schema.ipynb`](labs/D1_2_improve_prompt_schema.ipynb) |
| 1.3 | Day 1 · Session 3 | Compare licence and acceptable-use terms of three small models and record which may be adapted and deployed | [`D1_3_compare_licence_acceptable.ipynb`](labs/D1_3_compare_licence_acceptable.ipynb) |
| 1.4 | Day 1 · Session 4 | Estimate tuning, serving and per-request cost for a large hosted model versus a tuned small model on-premises | [`D1_4_estimate_tuning_serving.ipynb`](labs/D1_4_estimate_tuning_serving.ipynb) |
| 2.1 | Day 2 · Session 1 | Turn 500 synthetic traces and staff corrections into chat-format examples | [`D2_1_turn_500_synthetic.ipynb`](labs/D2_1_turn_500_synthetic.ipynb) |
| 2.2 | Day 2 · Session 2 | Generate 300 examples with a larger model and filter them with schema and rule checks | [`D2_2_generate_300_examples.ipynb`](labs/D2_2_generate_300_examples.ipynb) |
| 2.3 | Day 2 · Session 3 | Render examples with the model's template and fix three formatting faults that would harm training | [`D2_3_render_examples_model.ipynb`](labs/D2_3_render_examples_model.ipynb) |
| 2.4 | Day 2 · Session 4 | Remove near-duplicates and test-set overlap and scan the set for personal data | [`D2_4_remove_near_duplicates.ipynb`](labs/D2_4_remove_near_duplicates.ipynb) |
| 3.1 | Day 3 · Session 1 | Train an adapter on the supplied set in a free notebook GPU and read the loss curves | [`D3_1_train_adapter_supplied.ipynb`](labs/D3_1_train_adapter_supplied.ipynb) |
| 3.2 | Day 3 · Session 2 | Train the same adapter on a 4-bit base and compare memory, time and score with a supplied 16-bit run | [`D3_2_train_same_adapter.ipynb`](labs/D3_2_train_same_adapter.ipynb) |
| 3.3 | Day 3 · Session 3 | Read a supplied sweep of rank and learning rate, run one more setting and choose on the validation set | [`D3_3_read_supplied_sweep.ipynb`](labs/D3_3_read_supplied_sweep.ipynb) |
| 3.4 | Day 3 · Session 4 | Run a short distillation of the larger model's filtered tool calls into the student and compare it with a supplied corrections-only adapter | [`D3_4_run_short_distillation.ipynb`](labs/D3_4_run_short_distillation.ipynb) |
| 4.1 | Day 4 · Session 1 | Compare tuned, base and larger prompted models on Arabic and English slices with a paired bootstrap | [`D4_1_compare_tuned_base.ipynb`](labs/D4_1_compare_tuned_base.ipynb) |
| 4.2 | Day 4 · Session 2 | Run general-ability and refusal checks on a supplied adapter and list what got worse | [`D4_2_run_general_ability.ipynb`](labs/D4_2_run_general_ability.ipynb) |
| 4.3 | Day 4 · Session 3 | Serve a quantised base with two adapters and load-test throughput | [`D4_3_serve_quantised_base.ipynb`](labs/D4_3_serve_quantised_base.ipynb) |
| 4.4 | Day 4 · Session 4 | Draft the dataset card, model card and decision memo, then choose the capstone task | [`D4_4_draft_dataset_card.ipynb`](labs/D4_4_draft_dataset_card.ipynb) |
<!-- /academy:labs -->

## Capstone

An adapted small model for a task of the participant's choice, submitted as a public repository with the dataset card, evaluation against the base and a larger prompted model, serving cost and decision memo, followed by a ten-minute defence

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

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

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- NDMO Data Classification Policy: REPLACE: the address of the official text
- SDAIA Generative AI Guidelines for Government: REPLACE: the address of the official text
- SDAIA AI Ethics Principles: REPLACE: the address of the official text
