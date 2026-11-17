---
course: "Model Adaptation: Fine-Tuning, Distillation and Small Models"
code: "AE-310"
version: "2027.1.0"
deck: "capstone"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-310/slides/capstone.html"
---

# Model Adaptation: Fine-Tuning, Distillation and Small Models

## Capstone: Capstone: an adapted small model and its defence

Ten minutes each: the decision memo, a live re-run of the paired comparison and one regression explained

#### [capstone#title] Capstone: an adapted small model and its defence

*SDAIA Academy · Model Adaptation: Fine-Tuning, Distillation and Small Models*

Capstone: Ten minutes each: the decision memo, a live re-run of the paired comparison and one regression explained

AI Engineering Track · 4 teaching days + capstone day · 10 min talk and demo · AE-310

#### [capstone#agenda] How the presentations run

*Capstone*

- **Expectations and final prep**: the rubric, a last rehearsal, then the first presentations
- **Presentations**: each one timed, with questions after it
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#rubric] The rubric

*How presentations are scored*

| Item | Value |
| --- | --- |
| Adaptation decision | 15% |
| Data quality | 25% |
| Training | 15% |
| Evaluation | 30% |
| Serving and cost | 15% |

> **Evaluation carries the most weight.** Tuned against the base and against a larger prompted model, on the same held-out requests, with an interval and by slice.

#### [capstone#format] Ten minutes for each defence, questions included

*Format*

- **1 · The decision · 3 min.** The task, the failure classes, the frozen baseline, and what you decided: ship, ship with conditions, or do not ship.
- **2 · The evidence · 4 min.** Run the paired comparison again, live, from the repository. Then the slices, the cost per thousand requests and the licence line.
- **3 · One regression · 3 min.** The assessor picks one thing that got worse. Explain where it comes from and what you would do about it.

> **"Do not ship" is a full-marks answer** when the evidence supports it. The memo is assessed on its reasoning, not on the score of the model.

#### [capstone#repository] A public repository that someone else could run

*What the assessor opens*

- **Decision:** failure classes on the baseline, and the frozen prompt
- **Data:** the tuning set or the script that builds it, the dataset card, the overlap report
- **Training:** the adapter and its run log, with timestamps
- **Evaluation:** outputs of the three systems, the paired comparison, the regression check
- **Serving and cost:** the load-test report and the cost per thousand requests
- **The memo:** one page

- **Before you make it public.** No key, no real person's data, no data you may not publish, and a base model whose licence allows what you did. Synthetic or public data only.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Project |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first presentation*

Run the paired comparison once from a clean clone, and open the memo on this screen.

#### [capstone#now-presenting] Defence, questions included

*Now presenting*

#### [capstone#review] What the presentations showed

*Trainer review*

- **Strongest patterns.** What several projects got right.
- **Common gaps.** What most projects would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **The follow-up task:** run your paired comparison again on requests collected after today and add the result to the repository
- **Your project stays yours:** keep it reproducible, and act on the regression you were asked about
- **Next Academy course:** AE-406, Post-Training Open Models on Domain Data, takes up where one adapter on one accelerator ends

#### [capstone#thanks] Thank you

*SDAIA Academy*

Model Adaptation: Fine-Tuning, Distillation and Small Models
