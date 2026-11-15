# Model Adaptation: Fine-Tuning, Distillation and Small Models: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Prompt, retrieve, tune or distil | [`D1_1_run_small_reference.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D1_1_run_small_reference.ipynb) |
| Day 1 | Session 2 · The baseline tuning must beat | [`D1_2_improve_prompt_schema.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D1_2_improve_prompt_schema.ipynb) |
| Day 1 | Session 3 · Licences and terms of open-weight models | [`D1_3_compare_licence_acceptable.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D1_3_compare_licence_acceptable.ipynb) |
| Day 1 | Session 4 · Cost and placement of the options | [`D1_4_estimate_tuning_serving.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D1_4_estimate_tuning_serving.ipynb) |
| Day 2 | Session 1 · Tuning data from traces and corrections | [`D2_1_turn_500_synthetic.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D2_1_turn_500_synthetic.ipynb) |
| Day 2 | Session 2 · Teacher generation and verifier filtering | [`D2_2_generate_300_examples.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D2_2_generate_300_examples.ipynb) |
| Day 2 | Session 3 · Chat templates and tool-call formats | [`D2_3_render_examples_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D2_3_render_examples_model.ipynb) |
| Day 2 | Session 4 · Deduplication, decontamination and personal data | [`D2_4_remove_near_duplicates.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D2_4_remove_near_duplicates.ipynb) |
| Day 3 | Session 1 · Low-rank adapters | [`D3_1_train_adapter_supplied.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D3_1_train_adapter_supplied.ipynb) |
| Day 3 | Session 2 · Quantised adapters and memory | [`D3_2_train_same_adapter.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D3_2_train_same_adapter.ipynb) |
| Day 3 | Session 3 · Settings that matter | [`D3_3_read_supplied_sweep.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D3_3_read_supplied_sweep.ipynb) |
| Day 3 | Session 4 · Distilling a behaviour | [`D3_4_run_short_distillation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D3_4_run_short_distillation.ipynb) |
| Day 4 | Session 1 · Paired evaluation against the base | [`D4_1_compare_tuned_base.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D4_1_compare_tuned_base.ipynb) |
| Day 4 | Session 2 · Regressions and safety after tuning | [`D4_2_run_general_ability.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D4_2_run_general_ability.ipynb) |
| Day 4 | Session 3 · One base, several adapters | [`D4_3_serve_quantised_base.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D4_3_serve_quantised_base.ipynb) |
| Day 4 | Session 4 · Cards and the decision memo | [`D4_4_draft_dataset_card.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-310/blob/main/labs/D4_4_draft_dataset_card.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
