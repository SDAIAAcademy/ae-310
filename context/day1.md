---
course: "Model Adaptation: Fine-Tuning, Distillation and Small Models"
code: "AE-310"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-310/slides/day1.html"
---

# Model Adaptation: Fine-Tuning, Distillation and Small Models

## Day 1: Decide whether to adapt

Sort the failures, freeze the baseline, read the licences and count the cost before any training

#### [day1#title] Decide whether to adapt

*SDAIA Academy · Model Adaptation: Fine-Tuning, Distillation and Small Models*

Day 1: Sort the failures, freeze the baseline, read the licences and count the cost before any training

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-310

#### [day1#agenda] Decide whether to adapt

*Today*

- **Session 1 · Prompt, retrieve, tune or distil**: Four ways to change what a model does, and which failures each one can fix
- **Session 2 · The baseline tuning must beat**: A weak prompt flatters tuning: improve the prompt first, then freeze it with its score
- **Session 3 · Licences and terms of open-weight models**: Open weights are not one licence: what you may adapt, deploy and distribute depends on the text
- **Session 4 · Cost and placement of the options**: What each option costs per thousand requests, and where the data is processed

#### [day1#five-days] Decide, build the data, train, prove it, then defend the decision

*The course in one line per day*

1. **Day 1**: Decide whether to adapt (failures, baseline, licence, cost)
2. **Day 2**: Data for tuning (traces, teacher, templates, cleaning)
3. **Day 3**: Train adapters and distil (16-bit, 4-bit, settings, a behaviour)
4. **Day 4**: Evaluate and serve (paired tests, regressions, load)
5. **Day 5**: Capstone (your own adapted model, defended)

#### [day1#case-al-bayraq] Al-Bayraq Chamber of Commerce and its assistant, Qayd

*The case study · fictional, synthetic data*

- Members write in **Arabic and English**: new memberships, renewals, record changes, certificates
- **Qayd** turns each request into one call to the registry
- Requests carry personal data and are classified **Restricted**: they stay on the chamber's own servers
- So Qayd runs on a **small open-weight model**, and prompted alone it fails too often

- **The people you work for.** **Maysoon Al-Dabbagh** signs the decision. **Khuzama Al-Rumaizan**'s registry desk corrects Qayd. **Jarrah Al-Suhaim** owns the one accelerator. **Anoud Al-Fuhaid** protects the data, **Zakariya Al-Mousa** reads the licences. You are **Dhay Al-Shuwaier**, the engineer.

#### [day1#setup-check] Setup check: one notebook proves your environment

*Before session 1*

- Open `labs/00_setup_check.ipynb`, in Colab or locally with `uv`
- It imports every library, reads one data file and loads the course model
- With a key, it asks each hosted model one question
- Without a key, a **scripted stand-in** answers, and every lab says so on its first line

- **Two kinds of model in this course.** Days 1 and 2 **call** free hosted models. Days 3 and 4 **train**: the course's own small model on any processor, or an open-weight model on a notebook GPU.

### Session 1: Prompt, retrieve, tune or distil

#### [day1#s1] Prompt, retrieve, tune or distil

*Session 1*

- Name what prompting, retrieval, tuning and distillation each change in a system
- Sort the failures of a prompted model into knowledge, format and behaviour
- Assign each failure class to an approach, with counts

#### [day1#four-levers] Each lever changes a different part of the system

*Four ways to change what a model does*

- **Prompt.** Changes the **instructions**. Minutes to try, read again on every request.
- **Retrieve.** Changes what the model **is shown**: a table, a document, a record.
- **Tune.** Changes the **weights**, from your examples. Days of work, then no extra tokens.
- **Distil.** Tuning whose examples come from a **larger model**.

> **Order of cost.** Try them left to right. Each one you skip is a cheaper fix you never tested.

#### [day1#behaviour-not-knowledge] Tuning changes behaviour more than knowledge

*What tuning is good at*

- **Tuning teaches well.**
  - a format, exactly and every time
  - which tool to choose
  - when to stop and hand over to a person
  - a register, a language, a house style
- **Tuning teaches badly.**
  - facts that change: a code table, a fee, a deadline
  - facts seen once in the training set
  - anything you must be able to correct tomorrow

A fact in the weights cannot be edited, cited or dated. A fact in the context can.

#### [day1#narrow-tasks] Small models suit narrow tasks, not multi-step work

*Where a small model belongs*

| A small tuned model fits | It does not fit |
| --- | --- |
| One request in, one structured answer out | A plan of several steps that depend on each other |
| A fixed set of tools and fields | Tools it has never seen |
| Thousands of similar requests | Rare requests that each need judgement |
| An answer a program can check | An answer only an expert can check |

> **Qayd is on the left.** One request, one of five tools, fields a schema can verify. The rare and the odd go to the registry desk through `escalate_to_staff`.

#### [day1#request-to-call] One request in, exactly one tool call out

*The task, once*

- member: We moved our head office to Abha. Please update the record of CR 4064749760. Our partner company holds CR 4071506418; do not change their record.
- tool call: `{"name": "update_record", "arguments": {"cr_number": "4064749760", "field": "city", "value": "Abha"}}`
- graded by code: the same tool and the same arguments as the correct call, or it is wrong

Five tools: `create_membership`, `renew_membership`, `update_record`, `issue_certificate`, `escalate_to_staff`.

#### [day1#three-failure-classes] Three classes, because three different levers fix them

*Sorting failures*

| Class | What you see | Example in Qayd's answers |
| --- | --- | --- |
| **Knowledge** | Right call, wrong fact the model was never given | `"activity_code": "4711"` for information technology |
| **Format** | No call that parses, or one the interface rejects | a sentence, a code fence, a Python dictionary, digits left as ٤٠… |
| **Behaviour** | A valid call that is the wrong decision | renews with an invented number where it should escalate |

#### [day1#larger-model-instrument] The larger model is an instrument: it shows what capability can fix

*Why run a larger model you will not deploy*

| Item | Value |
| --- | --- |
| Small model, right | 27 |
| Larger model, right | 63 |
| Larger model, knowledge failures | 23 |

Lab 1.1, 100 requests, instructions version 0. Stand-in run.

- Where the larger model **also fails**, capability is not the problem: give the model the fact
- Where it **succeeds**, the behaviour exists and can be taught to the small model
- Where both fail on behaviour, look at the task before the model

#### [day1#class-to-lever] The decision table assigns each class to a lever, with counts

*The evidence for outcome 1*

| Class | Small | Larger | Approach |
| --- | --- | --- | --- |
| Knowledge | 10 | 23 | Retrieval: put the code table in the context |
| Format | 37 | 9 | Prompting first; tuning for what remains |
| Behaviour | 26 | 5 | Tuning, or distillation from the larger model |

> **Read the knowledge row twice.** The rule sorts format first, so the small model's knowledge failures hide behind its format failures. Fix the format and they appear.

#### [day1#check-failure-class] Which lever?

*Checkpoint*

**Question:** Qayd writes valid calls, but for "building materials" it answers with activity code `G-46`. The larger model, with the same instructions, makes the same mistake. What do you try first?

- A. Tune the small model on examples that contain the codes
- B. Replace the small model with a larger one
- C. Put the chamber's code table in the context
- D. Add a stricter output schema

#### [day1#lab-1-1] Sort the failures

Lab 1.1 · 25 min

Run the small model and the larger reference model on 100 requests, grade every answer by code, sort the failures into knowledge, format and behaviour, and fill the decision table. Assistants are off in this lab.

[D1_1_failure_sort.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D1_1_failure_sort.ipynb)

- [ ] Run both models on 100 requests (4 min)
- [ ] Grade by code and read ten failures (6 min)
- [ ] Write the rule that sorts a failure (8 min)
- [ ] Fill the decision table (7 min)
- [ ] ✱ Stretch: Split the behaviour failures by language: is one language carrying them?

### Session 2: The baseline tuning must beat

#### [day1#s2] The baseline tuning must beat

*Session 2*

- Improve a prompt with schema hints, context and examples, one change at a time
- Report a score on 100 requests with an interval
- Freeze a baseline: the prompt, the set and the score

#### [day1#weak-baseline-flatters] A weak prompt flatters tuning

*The most common false result in fine-tuning*

- "Our tuned model went from 27 to 85" compares a week of work with **one careless sentence**
- Half of that gain may be available in an afternoon, with no training at all
- The honest comparison is against the **best prompt you could write**

- **What the baseline is for.** It is the number the adapter must beat on Day 4. If the adapter cannot beat a good prompt on the same model, the project ends here, cheaply.

#### [day1#one-change-at-a-time] One change at a time, each scored on the same 100 requests

*Method*

**v0** one sentence → **v1** schema hints → **v2** the code table → **v3** the rules → **v4** four examples

> **Each change targets a class.** Schema hints: format. The table: knowledge. Rules and examples: behaviour, as far as words can reach it.

#### [day1#schema-hint] A schema hint tells the model what a call looks like

*Version 1*

Prompt for prompts.SCHEMA_HINT:

```text
Answer with exactly one tool call and nothing else, in this form: <tool_call> {"name": "<tool>", "arguments": {...}} </tool_call>
Tools: create_membership(company_name, cr_number, city, activity_code) … escalate_to_staff(reason)
cr_number is 10 Western digits (convert Arabic-Indic digits). company_name is copied as written. city is the English name.
```

The wrapper · The interface · How to write each field

#### [day1#examples-from-validation] Examples come from validation requests, never from the test set

*Version 4*

- **Validation set.** 120 requests. The source of the four examples, and of every choice on Day 3.
- **The 100 requests.** Analysis and the baseline score. An example taken from here is an answer shown in advance.
- **Test set.** 400 requests from companies no other set contains. Opened on Day 4, once.

> **No personal data in an example.** An example is sent with every request, for as long as the prompt lives.

#### [day1#prompt-scores] Words alone took the small model from 27 to 67 in 100

*Lab 1.2 · stand-in run*

| Item | Value |
| --- | --- |
| v0 · one sentence | 27 |
| v1 · schema hints | 35 |
| v2 · code table | 50 |
| v3 · rules | 52 |
| v4 · four examples | 67 |
| Larger model, v4 | 94 |

#### [day1#interval-on-100] A score on 100 requests carries an interval of about ±10 points

*How sure is 67 out of 100*

**57 to 75** 95% Wilson interval for 67 right out of 100

- v2 scored 50 and v3 scored 52: two requests apart, well inside the noise
- The interval answers "what would another hundred requests give?"
- It does not compare two versions: the versions ran on the **same** requests, and Day 4 uses that pairing

#### [day1#frozen-baseline] A baseline is three things that never change again

*What "frozen" means*

- **The instructions.** Identified by a hash of their text and examples. One edited sentence is a new baseline.
- **The set.** The same file of requests, by name and by content.
- **The score.** Right answers, n, and the interval. Recorded with the model's name and the day.

```json
// file: artifacts/1.2/baseline.json
{"prompt_sha256": "bdf86a63…", "n": 100, "correct": 67,
 "accuracy": 0.67, "interval": [0.573, 0.754],
 "set": "data/requests_100.jsonl"}
```

#### [day1#what-remains] What the best prompt leaves behind is behaviour

*The case for tuning, stated honestly*

- The schema is there, the table is there, the rules are written
- The model still takes the partner's number, still renews where it should escalate
- The larger model, same words: **94**. Room above the baseline: **27 requests in 100**

> **Now tuning has a target.** Not "make it better": remove the behaviour failures that 33 requests in 100 still show, and reach towards 94.

#### [day1#check-baseline] Better than what?

*Checkpoint*

**Question:** A colleague reports: "Our adapter scores 84 on the 100 requests; the prompted model scored 27." The 27 is version 0. What does the report show?

- A. Less than it claims: the adapter must be compared with the frozen baseline, 67
- B. That tuning gained 57 points
- C. Nothing, because 100 requests are too few to measure anything
- D. That the adapter should be compared with the larger model only

#### [day1#lab-1-2] Freeze the baseline

Lab 1.2 · 25 min

Score five versions of the instructions on the same 100 requests, one change at a time, and freeze the best with its hash, its score and its interval as the baseline that tuning must beat.

[D1_2_prompt_baseline.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D1_2_prompt_baseline.ipynb)

- [ ] Score version 0 with an interval (4 min)
- [ ] Add schema hints (5 min)
- [ ] Add the code table and the rules (6 min)
- [ ] Add four examples (5 min)
- [ ] Freeze the baseline and list what remains (5 min)
- [ ] ✱ Stretch: Run the frozen instructions on the larger model: how much room is left above the baseline?

### Session 3: Licences and terms of open-weight models

#### [day1#s3] Licences and terms of open-weight models

*Session 3*

- Distinguish permission to adapt, to deploy and to distribute in a model licence
- Find conditions that depend on who deploys and for what
- Record which candidate models the chamber may adapt and deploy

#### [day1#open-weights-not-one-licence] "Open weights" means you can download it, not what you may do with it

*Before any training*

- Every model on the Hub has a licence file. The **tag** on the card is a label someone typed
- Licences differ on purpose, on revenue, on field of use and on what you pass on
- The check costs an hour **before** training and a project **after** deployment

- **Who decides.** You find the clause and the fact it depends on. **Zakariya Al-Mousa**, the chamber's counsel, decides. "Ask counsel" is a valid entry in your table.

#### [day1#three-permissions] Adapt, deploy and distribute are three separate permissions

*Read a licence with three questions*

- **Adapt.** Fine-tune it. The result is a **derivative work** in most licences.
- **Deploy.** Run it in a service. Here licences ask **for what purpose** and **by whom**.
- **Distribute.** Give the weights or the adapter to someone else. Conditions travel with it.

> **The chamber's plan needs all three:** tune on its requests, run inside a service members pay fees for, give the adapter to two sister chambers.

#### [day1#three-licence-families] Three candidate models, three kinds of licence

*Lab 1.3 · read on 15 November 2026*

| Model | Licence | The clause that decides |
| --- | --- | --- |
| Qwen3-0.6B | Apache License 2.0 | Grants derivative works and redistribution; no clause on purpose or revenue |
| LFM3-1.7B | LFM Open License v1.0 | Commercial use is licensed only below USD 10 million of annual revenue |
| Command R7B Arabic | CC BY-NC 4.0, gated | Every right is for non-commercial purposes; more conditions may sit behind the gate |

Names are examples, not endorsements. The slot matters: a permissive licence, one that depends on who you are, one that depends on what for.

#### [day1#conditions-need-facts] Two conditions turn on facts about the chamber

*A licence table without facts is an opinion*

- **The revenue threshold.** The chamber's annual revenue is SAR 62 million, which is USD 16.5 million: **above** the threshold. Its use is licensed only if it is not "commercial use".
- **The purpose.** The chamber is a non-profit body, and its service charges fees. Is that "primarily intended for commercial advantage or monetary compensation"? The text does not say.

> **Write the question down now.** "Ask" in the table, with the clause and the fact, is a result. Finding out after deployment is not.

#### [day1#tag-is-not-text] Read the text, record the day, look for a second document

*Three habits*

- **The text, not the tag.** Open the licence file in the repository; a card can say "other" and hide everything
- **The day you read it.** A later release of the same family can carry different terms
- **A second document.** An acceptable-use policy, or conditions on a gated page, bind you as well
- **The teacher too.** On Day 3 a larger model writes your training labels: its terms on using outputs to train another model belong in the same table

#### [day1#licence-line] The decision memo carries one licence line

*What Maysoon reads*

**Weak (weak example):** All three models are open source, so licensing is not a concern.

**Usable (good example):** Take Qwen3-0.6B forward: Apache License 2.0, sections 2 and 4, allows adapting, deploying and passing on the adapter with the licence text and a notice of changes. LFM3-1.7B needs counsel's answer on section 5: the chamber is above the revenue threshold.

#### [day1#check-licence] May we deploy it?

*Checkpoint*

**Question:** A model's licence allows commercial use only to organisations with annual revenue under USD 10 million. The chamber's revenue is above that, and its registration service charges fees. What goes in the "deploy" column?

- A. Yes: the weights are open, so use is free
- B. Yes: a chamber is a non-profit body, so the limit cannot apply
- C. Yes, once the model is fine-tuned, because the adapter is ours
- D. Ask: counsel must decide whether a fee-charging service is commercial use

#### [day1#lab-1-3] Three licences

Lab 1.3 · 25 min

Read the licence sheets of three small models, check the chamber's facts against each condition, and record which models may be adapted and deployed, with the clause that decides.

[D1_3_licences.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D1_3_licences.ipynb)

- [ ] Read three licence sheets (6 min)
- [ ] Record adapt, deploy and distribute for each (8 min)
- [ ] Check the chamber's facts against each condition (6 min)
- [ ] Write the licence line of the memo (5 min)
- [ ] ✱ Stretch: Fill a sheet for a fourth model of your choice from its licence text on the Hub

### Session 4: Cost and placement of the options

#### [day1#s4] Cost and placement of the options

*Session 4*

- Estimate tokens and cost per request from measured requests
- Compare a per-token price with an owned accelerator at a given volume
- State where the data is processed under each option

#### [day1#two-cost-shapes] A hosted model costs per token; an owned accelerator costs per month

*Two ways to pay for a model*

- **Hosted.**
  - a price per million input and output tokens
  - cost grows with volume, starts at zero
  - the request leaves your servers
- **Owned.**
  - purchase over its useful life, power, an engineer's hours
  - the same cost at one request or a million
  - the request stays where you put the machine

> **Cost per thousand requests** is the unit that makes them comparable, and for the owned machine it depends entirely on volume.

#### [day1#tokens-per-request] A prompted request is mostly instructions

*Lab 1.4 · measured on the 100 requests*

- **623** input tokens, prompted: instructions, table, rules, examples and the request
- **30** input tokens, tuned: the request alone
- **33** output tokens: one tool call

Counts are the lab's rough estimate (four characters of English, two and a half of Arabic to a token). A provider's own count replaces them.

#### [day1#cost-two-options] At the chamber's volume the hosted model is far cheaper

*Lab 1.4 · prices read on 15 November 2026*

| Option | SAR per 1,000 requests | Where a request is processed |
| --- | --- | --- |
| Large hosted model | 0.52 | The provider's servers |
| Tuned small model, own accelerator | 141.38 | The chamber's own servers |

> **Do not soften this.** SAR 5,655 a month for the machine, 40,000 requests a month. Break-even is about 11 million requests a month: 274 times the chamber's volume.

#### [day1#placement-decides] Cost argues for the hosted model; placement rules it out

*What actually decides*

- A request names an owner, a mobile number, sometimes an identity number: **personal data**
- The chamber classifies registration requests as **Restricted**
- Its requirement: they are processed on its **own servers**

- **The honest sentence for the memo.** "The chamber pays a premium of about SAR 141 per thousand requests to keep members' data on its own servers. This is the premium, and this is why it is worth paying."

#### [day1#tuning-is-one-off] Tuning is paid once; serving is paid every month

*Three costs, three shapes*

| Cost | Shape | In the case |
| --- | --- | --- |
| Tuning | Once per version of the adapter | SAR 13,260: 60 engineer hours and 10 GPU hours |
| Serving | Every month, whatever the volume | SAR 5,655 |
| Per request | Serving divided by volume | SAR 0.14 at 40,000 requests a month |

> **Tuning is paid back in quality, not money.** A prompted and a tuned small model cost the same to serve on the same machine. And the engineer costs more than the GPU.

#### [day1#check-cost] The volume doubles

*Checkpoint*

**Question:** The chamber's requests double to 80,000 a month. The accelerator still has spare capacity. What happens to the cost per thousand requests?

- A. Both options stay the same
- B. Hosted stays the same; the owned accelerator's halves
- C. Hosted halves; the owned accelerator's stays the same
- D. Both double

#### [day1#lab-1-4] Cost of two options

Lab 1.4 · 25 min

Estimate tuning, serving and per-request cost for a large hosted model and for a tuned small model on the chamber's own accelerator, and find the volume at which they break even.

[D1_4_cost_placement.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D1_4_cost_placement.ipynb)

- [ ] Measure tokens per request (5 min)
- [ ] Cost per thousand requests on a hosted model (5 min)
- [ ] Cost per thousand on an owned accelerator (7 min)
- [ ] Break-even volume and placement (8 min)
- [ ] ✱ Stretch: Add the one-off cost of tuning and find the month in which it is paid back

#### [day1#wrap-up] Three things to keep

*Day 1*

- **Sort before you train.** Knowledge goes to the context, format to the prompt first, behaviour to tuning or distillation. With counts.
- **Freeze a strong baseline.** The adapter is compared with the best prompt on the same model, never with the first one.
- **Licence and cost come first.** Read the text before training. State what the option costs and why it is chosen anyway.

#### [day1#next] Day 2 · Data for tuning

*Tomorrow*

500 pilot traces, the registry desk's corrections and a larger model's labels become one clean tuning set.

- Guided self-study before tomorrow: read your model's chat template on its model card, and find where a tool call goes
- Keep today's four files in `labs/artifacts/`; tomorrow's labs start from supplied files either way
