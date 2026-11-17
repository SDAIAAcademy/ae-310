---
course: "Model Adaptation: Fine-Tuning, Distillation and Small Models"
code: "AE-310"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-310/slides/day3.html"
---

# Model Adaptation: Fine-Tuning, Distillation and Small Models

## Day 3: Train adapters and distil

Train low-rank adapters on 16-bit and 4-bit bases, choose settings on validation and distil a behaviour

#### [day3#title] Train adapters and distil

*SDAIA Academy · Model Adaptation: Fine-Tuning, Distillation and Small Models*

Day 3: Train low-rank adapters on 16-bit and 4-bit bases, choose settings on validation and distil a behaviour

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-310

#### [day3#agenda] Train adapters and distil

*Today*

- **Session 1 · Low-rank adapters**: A library adapter on a decoder: what trains, what stays frozen and what the curves say
- **Session 2 · Quantised adapters and memory**: Where training memory goes, and what a 4-bit base under 16-bit adapters costs
- **Session 3 · Settings that matter**: Rank, scaling, learning rate and schedule: which settings move the score, read from a sweep
- **Session 4 · Distilling a behaviour**: A behaviour is a policy: teach the student what the larger model does, errors included

#### [day3#review-day2] What must be true of the text before the first training step?

*Review of Day 2*

**Question:** You are about to train on 661 examples. Which check from yesterday can no longer be made once the run has finished and the loss has fallen?

- A. That the tuned model beats the base on the test set
- B. That the validation loss fell
- C. That the rendered text ends the turn and matches the model's template
- D. That the adapter has the intended rank

#### [day3#two-tiers] One code path, two base models

*How today runs*

- **On a notebook GPU.** A small open-weight model from the Hub, Qwen3-0.6B, in 16 bits or in 4 bits. Downloaded into the notebook, not to your laptop. Free GPUs vary in type and limits.
- **On a processor.** The course model in the repository: the same layout at about one million parameters, with its own tokeniser and template. A run takes a few minutes.

> **The supplied runs and all numbers on the slides are the course model's.** It was pretrained by the course to stand where a small open-weight model stands in this case: able in general, refusing requests for personal data, and answering registration requests in an older text format.

### Session 1: Low-rank adapters

#### [day3#s1] Low-rank adapters

*Session 1*

- Configure a low-rank adapter: rank, scaling and target modules
- Train an adapter with a short schedule and log the run
- Read training and validation loss curves and name what they show

#### [day3#what-the-base-does] The base knows the task and writes it in the wrong form

*Before any training*

**request:** Request: certificate of membership. CR number 4042480051.

**base (weak example):** CALL issue_certificate cr_number=4042480051 kind=membership

**wanted:** \<tool_call> {"name": "issue_certificate", "arguments": {"cr_number": "4042480051", "kind": "membership"}} \</tool_call>

On the 120 validation requests the base produces **0 valid calls**. In its old format it has the content right on 85 of the 101 requests that need a registry call.

#### [day3#full-tuning-cost] Full fine-tuning moves every weight, and keeps a copy per task

*Why not train everything?*

- Gradients and optimiser state for every parameter: several times the memory of the weights
- One full copy of the model for each task you tune
- Every weight can move, so everything the base knew can be disturbed

- **The adapter's bet.** The change a task needs is small and has structure. Freeze the base; learn only a low-rank correction to a few of its matrices.

#### [day3#lora-idea] W stays frozen; the adapter adds B·A beside it

*Low-rank adaptation*

1. **W** d_out × d_in, frozen
2. **B** d_out × r, starts at zero
3. **A** r × d_in, starts random
4. **alpha / r** a fixed scale

- Parameters added to one matrix: **r × (d_in + d_out)**
- B starts at zero, so at step 0 the adapted model **is** the base
- Which matrices get an adapter is a setting: today, the query and value projections of every attention layer

#### [day3#what-trains] 1.6% of the parameters train

*Lab 3.1 · task 1*

- **1,001,728** parameters in the course model, all frozen
- **16,384** in a rank-8 adapter on 8 matrices
- **68 KB** the adapter on disk: the only file you ship per task

> **Count before you train.** If the trainable share is not what you expect, the adapter is on the wrong modules, or the base is not frozen.

#### [day3#training-loop] The whole training loop fits on a slide

*tuning.train, in essence*

```python
# file: tuning.py
params = [p for p in model.parameters() if p.requires_grad]      # the adapter only
opt = torch.optim.AdamW(params, lr=lr)
for step in range(steps):
    x, labels, mask = next_batch()               # rendered with the template; labels masked on the prompt
    for group in opt.param_groups:
        group["lr"] = lr * schedule(step)        # warm-up, then cosine decay
    loss = model(input_ids=x, attention_mask=mask, labels=labels).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); opt.zero_grad()
    if (step + 1) % 25 == 0:
        log(step, loss, loss_on(model, validation))
```

#### [day3#two-curves] Two curves: one on what it trained on, one on what it did not

*Lab 3.1 · task 4*

| Step | Training loss | Validation loss |
| --- | --- | --- |
| 25 | 0.070 | 0.032 |
| 50 | 0.041 | 0.020 |
| 100 | 0.044 | 0.016 |
| 150 | 0.029 | 0.014 |
| 200 | 0.031 | 0.012 |
| 250 | 0.034 | 0.013 |
| 300 | 0.028 | 0.012 |

- **Both fall together:** the adapter is learning the task
- **Training falls, validation rises:** it is memorising the set. Stop earlier, or look for near-copies
- **Neither falls:** the learning rate, the label mask or the frozen flags

#### [day3#loss-is-not-score] A loss is not a score: grade the calls

*Lab 3.1 · task 5*

| Item | Value |
| --- | --- |
| Base, right | 0 of 120 |
| Adapter, valid calls | 92 of 120 |
| Adapter, right | 85 of 120 |

The registry accepts a call or it does not. A lower loss means the right tokens became more likely on average; one wrong digit in a registration number is a small loss and a rejected call.

#### [day3#run-log] An adapter without its run log cannot be reproduced

*What a run leaves behind*

```json
// file: adapters/registry-16bit/run_log.json
{"tier": "tiny", "steps": 300, "lr": 0.002, "batch": 8, "schedule": "cosine", "seed": 310,
 "train_rows": 661, "val_rows": 120, "trainable": 16384, "seconds": 115,
 "curve": [{"step": 25, "train_loss": …, "val_loss": …}, …],
 "val": {"right": 85, "of": 120}}
```

- Beside it: the base and its precision, the tuning set (the dataset card), the code version
- The capstone asks for run logs with timestamps in the repository

#### [day3#check-adapter] How many parameters train?

*Checkpoint*

**Question:** A base has 24 layers. You put a rank-8 adapter on the query and value projections of each layer; each is a 1,024 × 1,024 matrix. How many parameters train?

- A. 16,384
- B. 32,768
- C. 786,432
- D. 50,331,648

#### [day3#lab-3-1] Train an adapter, read the curves

Lab 3.1 · 25 min

Train a low-rank adapter on the supplied tuning set, read its training and validation loss, score it against the base on validation requests, and save the adapter with its run log.

[D3_1_lora_adapter.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D3_1_lora_adapter.ipynb)

- [ ] Load the base and count what trains (4 min)
- [ ] Score the base on validation requests (4 min)
- [ ] Train the adapter (8 min)
- [ ] Read the two loss curves (5 min)
- [ ] Score the adapter and save it with its log (4 min)
- [ ] ✱ Stretch: Read the supplied log of a run that went wrong and name the setting

### Session 2: Quantised adapters and memory

#### [day3#s2] Quantised adapters and memory

*Session 2*

- Estimate memory from parameter counts and precision
- Train an adapter on a 4-bit base
- Compare memory, time and score of a 16-bit and a 4-bit run

#### [day3#memory-arithmetic] Whether a model fits is arithmetic: parameters times bits

*Do this before you rent anything*

| Parameters | 16 bits | 4 bits |
| --- | --- | --- |
| 0.6 billion | 1.1 GB | 0.3 GB |
| 1.7 billion | 3.2 GB | 0.8 GB |
| 7 billion | 13.0 GB | 3.5 GB |
| 27 billion | 50.3 GB | 13.4 GB |

Weights only. 16 bits is two bytes a weight. 4 bits is half a byte, plus here one 16-bit scale for every block of 64 weights.

#### [day3#weights-are-the-floor] The weights are the floor; training adds three things on top

*Why "it fits" is not the end of it*

- **Activations.** Kept for every token of every example in the batch, at every layer, for the backward pass. Grows with batch size × sequence length.
- **Adapter gradients.** Small: only A and B have them.
- **Optimiser state.** Two more numbers for each trainable parameter with AdamW. Small for the same reason.

> **Out of memory with weights that fit?** Lower the batch size or the maximum length first. Gradient accumulation keeps the effective batch.

#### [day3#what-4-bit-means] A 4-bit weight is one of 16 levels, times the scale of its block

*Lab 3.2 · task 1*

```python
# file: Lab 3.2
scale   = block.abs().max()                          # one number kept per block of 64 weights
nearest = (block / scale - levels).abs().argmin()     # index of the closest of 16 levels: 4 bits
rounded = levels[nearest] * scale                     # what the model computes with
```

- The 16 levels are not evenly spaced: they are denser near zero, where most weights of a trained network lie (the NF4 type)
- Rounding is an error added to every weight. Whether it matters is an empirical question, per model and per task

#### [day3#qlora-arrangement] The base is rounded and frozen; the adapter trains in full precision

*Training on a quantised base*

**4-bit base** frozen, rounded → **Adapter** full precision, trainable → **Loss** on the assistant's turn

- Gradients pass through the rounded weights to the adapter; the rounded weights never change
- The adapter learns the task **and** part of what rounding lost
- So the adapter belongs to its base at that precision: serve it on the base it was trained on

#### [day3#sixteen-against-four] The same adapter on two bases

*Lab 3.2 · task 4*

|  | 16-bit base | 4-bit base |
| --- | --- | --- |
| Base weights (course model) | 1.91 MB | 0.85 MB |
| Untrained, valid calls of 120 | 0 | 0 |
| Final validation loss | 0.0124 | 0.0125 |
| Validation requests right, of 120 | 85 | 87 |

> **Why not 4 to 1?** The embedding table stays at 16 bits, and in a model this small it is a large share. In a model of billions of parameters the linear layers dominate and the ratio approaches the arithmetic.

#### [day3#check-memory] Will it fit?

*Checkpoint*

**Question:** An accelerator has 16 GB of memory. You want to train an adapter on a base with 7 billion parameters. What does the arithmetic say?

- A. Load it in 16 bits: 13 GB of weights fit in 16 GB
- B. Load it in 4 bits: about 3.5 GB of weights, leaving room for activations
- C. Either: only the adapter needs memory, and it is tiny
- D. It cannot be done on this accelerator

#### [day3#lab-3-2] The same adapter on a 4-bit base

Lab 3.2 · 25 min

Do the memory arithmetic for 16 and 4 bits, train the same adapter on a 4-bit base, and compare memory, time and score with the supplied 16-bit run.

[D3_2_four_bit.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D3_2_four_bit.ipynb)

- [ ] Memory arithmetic for 16 and 4 bits (6 min)
- [ ] Load the 4-bit base and score it untrained (4 min)
- [ ] Train the same adapter on it (8 min)
- [ ] Compare memory, time and score with the 16-bit run (7 min)
- [ ] ✱ Stretch: Load your 4-bit adapter on the 16-bit base and score it: does it transfer?

### Session 3: Settings that matter

#### [day3#s3] Settings that matter

*Session 3*

- Explain what rank, scaling, learning rate and schedule each control
- Read a sweep and separate a real difference from noise
- Choose a setting on the validation set, never on the test set

#### [day3#settings-ranked] A few settings decide the run; most can stay at a sensible value

*Where to spend your runs*

| Setting | What it does | Today |
| --- | --- | --- |
| **Learning rate** | The size of each step. Too low: nothing moves. Too high: the loss jumps or diverges | swept |
| **Steps** | How long. Read from the validation curve, not guessed | fixed at 300 |
| **Rank** | The capacity of the adapter, and its size | swept |
| Target modules | Which matrices get an adapter | query and value, fixed |
| Alpha | The scale of the update; tied to the rank | 2 × rank |
| Batch, warm-up, schedule | Stability of the steps | 8, 6%, cosine |

#### [day3#sweep-table] Eight supplied runs, one cell missing

*Lab 3.3 · task 1*

| Validation right, of 120 | lr 0.0002 | lr 0.002 | lr 0.02 | Trainable |
| --- | --- | --- | --- | --- |
| Rank 2 | 1 | 86 | 83 | 4,096 |
| Rank 8 | 21 | 85 | 86 | 16,384 |
| Rank 32 | 85 | **?** | 8 | 65,536 |

Read it by column, then by row. Which setting moves the score more?

#### [day3#noise-in-a-sweep] One run per cell, one seed: a few requests in 120 is not a difference

*How much of the table is real?*

- The same setting and seed, run twice here: 85 and 89 of 120
- A sweep read to the last digit chooses noise
- The honest table has several seeds per cell, or an interval

- **The lab's rule.** The highest validation score; among runs within 4 requests of the best, the one with the fewest trainable parameters. Chosen: **rank 2 at 0.002**, 86 of 120.

#### [day3#choose-on-validation] Choose on validation requests; the test set stays closed until Day 4

*The discipline*

1. **Tuning set** 661 · the adapter learns from it
2. **Validation** 120 · you choose settings on it
3. **Test** 400 · opened once, to report

> **Choosing is fitting.** A test set used to pick a setting has been fitted to, by you. Its score no longer says how the chosen model does on new requests.

#### [day3#rank-is-a-cost] Rank is a cost first, a quality setting second

*Lab 3.3 · stretch*

| Item | Value |
| --- | --- |
| Rank 2 · 4,096 parameters | best 86 of 120 |
| Rank 8 · 16,384 parameters | best 86 of 120 |
| Rank 32 · 65,536 parameters | best 88 of 120 |

Best score of each rank over the learning rates, the missing cell included. From rank 2 to rank 32: sixteen times the parameters for 2 more requests of 120, which is inside the noise.

#### [day3#budget-a-sweep] A sweep has a budget, and data usually beats it

*Before you launch one*

**15,625** runs in a grid of six settings at five values each

- Fix what one small sweep and the literature already tell you: targets, schedule, batch
- Sweep the learning rate first, then the rank
- Then spend the remaining hours on the data: one more source of corrections moves more than a fourth learning rate

#### [day3#check-sweep] Which run do you ship?

*Checkpoint*

**Question:** On 120 validation requests: rank 8 scores 101, rank 32 scores 102, rank 64 scores 100. Each is one run. Which do you choose, and on what ground?

- A. Rank 8: the three cannot be told apart, and it is the cheapest
- B. Rank 32: it has the highest score
- C. Rank 64: more capacity is safer
- D. Whichever scores best on the test set

#### [day3#lab-3-3] Read a sweep, add a run

Lab 3.3 · 25 min

Read the supplied sweep of rank and learning rate, predict and run the one setting it lacks, and choose a setting on the validation set with a reason.

[D3_3_sweep.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D3_3_sweep.ipynb)

- [ ] Read the supplied sweep (6 min)
- [ ] Predict the missing setting (3 min)
- [ ] Run it (9 min)
- [ ] Choose on the validation set and say why (7 min)
- [ ] ✱ Stretch: Plot validation accuracy against trainable parameters

### Session 4: Distilling a behaviour

#### [day3#s4] Distilling a behaviour

*Session 4*

- Distinguish distilling outputs from distilling logits
- Distil a larger model's tool-call behaviour into a small student
- Compare a distilled adapter with a corrections-only adapter on behaviour slices

#### [day3#what-distillation-is] Distillation: the student learns from the teacher's answers

*The fourth lever of Day 1*

1. **Teacher** a larger model, prompted
2. **Verifier** schema and rules (Lab 2.2)
3. **Filtered calls** 841 examples
4. **Student** the small model with an adapter

- Here the student sees only the teacher's final outputs: the same training as any other tuning set
- What is distilled is one **behaviour**, on one kind of request; the student does not become the teacher

#### [day3#two-sets-compared] Few labels from people, or many from a model

*Lab 3.4 · task 1*

|  | Staff corrections | Teacher, filtered |
| --- | --- | --- |
| Examples | 92 | 841 |
| Plain requests | 28 | 523 |
| Escalations | 27 | 107 |
| Two registration numbers | 21 | 109 |
| Arabic-Indic digits | 17 | 129 |
| Wrong labels (answer key) | 0 | 16 |

The corrections sit where Qayd failed, because that is why the desk corrected them. The teacher's set has the mix of real requests.

#### [day3#short-schedule] A short schedule is enough for one consistent behaviour

*Lab 3.4 · task 2*

**200 steps** on 841 teacher calls: validation loss 0.0282 → 0.0143 in 61 s

- Many consistent examples of the same decision: the student needs few passes
- Longer schedules buy memorisation of individual labels, the wrong ones included
- The course's promise holds: a short run on a free runtime

#### [day3#slices-compared] Compare by slice, and count the requests in each slice first

*Lab 3.4 · task 3*

| Validation slice | Requests | Corrections only | Distilled |
| --- | --- | --- | --- |
| Plain | 71 | 46 | 49 |
| Escalation | 19 | 18 | 19 |
| Two registration numbers | 12 | 4 | 6 |
| Arabic-Indic digits | 22 | 14 | 14 |
| All | 120 | 81 | 86 |

> **A slice of 12 requests moves by 8 points for each request.** The table says where to look, not how large a difference is.

#### [day3#student-copies] What the verifier let through, the student can learn

*Lab 3.4 · task 4*

- **16** wrong labels among the 841 kept (1.9%)
- **0** of those requests the student answers with the teacher's wrong call
- **12** it answers correctly although its label was wrong

> **A scattered error is outvoted. A systematic one is the lesson.** If the teacher is wrong every time on one kind of request, the student learns exactly that.

#### [day3#when-not-to-distil] Distillation needs a better teacher, usable outputs and data that may be sent

*Three questions before you distil*

- **Is the teacher better?** On this behaviour, measured on the same requests (Lab 1.1). A teacher that fails the same way has nothing to teach.
- **May its outputs train a model?** The licence of an open-weight teacher, or the terms of a hosted one. Read them as you read the student's licence on Day 1.
- **May the requests reach it?** Restricted requests may not go to a hosted teacher. Run the teacher on your own servers, or label synthetic and masked requests.

#### [day3#check-distil] Where do you fix it?

*Checkpoint*

**Question:** A distilled student takes the partner company's registration number in requests that name two. An audit shows the teacher does the same in about one such request in five. What do you change first?

- A. Train the student for more steps
- B. Raise the adapter's rank
- C. Lower the sampling temperature of the student
- D. The data: correct or remove those teacher labels, or add staff corrections for that kind of request

#### [day3#lab-3-4] Distil a behaviour

Lab 3.4 · 25 min

Train the student on the larger model's filtered tool calls with a short schedule and compare it, slice by slice, with the supplied adapter trained on staff corrections alone.

[D3_4_distil.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D3_4_distil.ipynb)

- [ ] Compare the two tuning sets (4 min)
- [ ] Train the student on the teacher's filtered calls (9 min)
- [ ] Score both adapters on the behaviour slices (7 min)
- [ ] Find a teacher error the student copied (5 min)
- [ ] ✱ Stretch: Train on corrections and teacher calls together and compare

#### [day3#wrap-up] Three things to keep

*Day 3*

- **Count, then train.** What trains, what fits in memory, and both curves. A run leaves a log or it did not happen.
- **Choose with a rule.** On validation requests, with a margin for noise, the cheapest among equals. The test set stays closed.
- **The student learns its labels.** The right ones and the wrong ones. Fix behaviour in the data.

#### [day3#next] Day 4 · Evaluate and serve

*Tomorrow*

The test set is opened once: tuned against base against the larger model, what got worse, and what it costs to serve.

- Guided self-study: write down, before tomorrow, the score you expect the tuned model to reach on the 400 test requests
- Bring a task of your own in mind for the capstone: one input, one checkable output
