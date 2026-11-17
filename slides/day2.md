---
course: "Model Adaptation: Fine-Tuning, Distillation and Small Models"
code: "AE-310"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-310/slides/day2.html"
---

# Model Adaptation: Fine-Tuning, Distillation and Small Models

## Day 2: Data for tuning

Build the tuning set from traces, corrections and a teacher, render it correctly and clean it

#### [day2#title] Data for tuning

*SDAIA Academy · Model Adaptation: Fine-Tuning, Distillation and Small Models*

Day 2: Build the tuning set from traces, corrections and a teacher, render it correctly and clean it

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-310

#### [day2#agenda] Data for tuning

*Today*

- **Session 1 · Tuning data from traces and corrections**: Pilot traces and staff corrections are labels with errors: keep what can be trusted
- **Session 2 · Teacher generation and verifier filtering**: A larger model labels what staff never saw, and a verifier decides what enters the set
- **Session 3 · Chat templates and tool-call formats**: The chat template is part of the model: train on the exact text the server will send
- **Session 4 · Deduplication, decontamination and personal data**: Duplicates inflate, leaks flatter and personal data does not belong in weights

#### [day2#review-day1] Which failures does today's data have to fix?

*Review of Day 1*

**Question:** With the frozen instructions, Qayd still takes a partner company's registration number in some requests. The code table and the schema are in the prompt. Which examples must today's tuning set contain?

- A. More rows of the activity code table
- B. Requests with two registration numbers, labelled with the applicant's own
- C. More requests that Qayd already answers correctly
- D. Examples of well-formed JSON

#### [day2#three-sources] Three sources become one clean tuning set

*The day in one picture*

1. **Traces** 500 pilot requests, 342 accepted
2. **Corrections** 123 by the registry desk
3. **Teacher** 300 labels by a larger model
4. **Clean** template, duplicates, leaks, personal data
5. **661 examples** with a card and an overlap report

Each lab starts from a supplied file, so the four can be done in any order and nobody waits for a neighbour.

### Session 1: Tuning data from traces and corrections

#### [day2#s1] Tuning data from traces and corrections

*Session 1*

- Read a trace and say which label, if any, it carries
- Join staff corrections to traces and reject corrections the interface refuses
- Write tuning examples in chat format with a tool call

#### [day2#trace-anatomy] A trace is a request, the model's answer and what a person did next

*The raw material*

- request: نريد التجديد لمدة سنة واحدة. سجلنا التجاري 4067281583.
- Qayd: `{"name": "renew_membership", "arguments": {"cr_number": "4067281583", "years": 2}}`
- registry desk: status: corrected
- correction: `{"name": "renew_membership", "arguments": {"cr_number": "4067281583", "years": 1}}`

The request asks for one year (سنة واحدة). Qayd wrote two. The desk's correction is the label.

#### [day2#three-statuses] Accepted, corrected, abandoned: three statuses, three kinds of label

*What label does a trace carry?*

| Status | Traces | The label | What it can teach |
| --- | --- | --- | --- |
| Accepted | 342 | Qayd's own output | The format, again. Nothing about its failures |
| Corrected | 123 | The desk's call | The behaviour Qayd lacks |
| Abandoned | 35 | None | Nothing, until someone labels it |

> **Abandoned traces are not random.** The desk gave up on the hardest requests. Leaving them out makes the tuning set easier than production.

#### [day2#accepted-is-not-correct] "Accepted" means nobody objected, not that the call was right

*The first trap*

**8 of 342** accepted traces carry a wrong call (the generator's answer key)

- A rule check finds **2** of them: the number in the call is not in the request
- The other 6 pass every check: valid, plausible, wrong
- In your own project there is no answer key: only an audit of a sample tells you the rate

#### [day2#corrections-have-errors] A correction is typed by a person at a busy desk

*The second trap*

| What the desk typed | What the registry says |
| --- | --- |
| `"cr_number": "403158269"` | not 10 Western digits |
| `"city": "الطائف"` | a value outside the list: the interface takes "Taif" |
| `"cr_number": "٤٠٥٠٥٢٢٦٨٧"` | not 10 Western digits |

> **Validate every label against the tool interface.** 12 of the 123 corrections are rejected. A label the tool would refuse is not a label.

#### [day2#validate-labels] The tool's own schema is the first reviewer of every label

*Lab 2.1 · task 3*

```python
# file: Lab 2.1
for trace in traces:
    call = labels[trace["id"]]          # Qayd's output, or the desk's call
    if call is None:
        continue                        # abandoned: no label
    errors = C.schema_errors(call)      # what the registry would refuse
    if errors:
        rejected.append((trace, call, errors))
    else:
        kept.append((trace, call))
```

#### [day2#chat-format-example] A tuning example is a short conversation with a tool call

*Lab 2.1 · task 4*

```json
// file: examples_traces.jsonl
{"id": "R-0412", "source": "correction", "lang": "ar",
 "messages": [
  {"role": "user", "content": "نريد التجديد لمدة سنة واحدة. سجلنا التجاري 4067281583."},
  {"role": "assistant", "content": "",
   "tool_calls": [{"type": "function", "function": {
       "name": "renew_membership",
       "arguments": {"cr_number": "4067281583", "years": 1}}}]}]}
```

Messages and `tool_calls`, not a rendered string: the model's template does the rendering (session 3). `source` stays with every row.

#### [day2#counts-are-the-card] What you dropped is the first paragraph of the dataset card

*Lab 2.1 · task 5*

| Item | Value |
| --- | --- |
| Kept from accepted traces | 342 |
| Kept from corrections | 111 |
| Abandoned, no label | 35 |
| Corrections rejected | 12 |

500 traces, 453 examples. Three quarters of them are what Qayd already does.

#### [day2#check-traces] Which rows enter the tuning set?

*Checkpoint*

**Question:** A corrected trace has the desk's call `{"name": "update_record", "arguments": {"cr_number": "4031582699", "field": "city", "value": "الدمام"}}`. The interface lists cities in English. What do you do with it?

- A. Keep it: a human correction is always right
- B. Change the value to "Dammam" and keep it as a correction
- C. Drop it and count it, or send it back to the desk
- D. Replace it with Qayd's original output

#### [day2#lab-2-1] Traces to examples

Lab 2.1 · 25 min

Turn 500 pilot traces and the registry desk's corrections into chat-format examples: keep accepted calls, apply corrections, validate every label against the tool interface and count what was dropped.

[D2_1_traces_corrections.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D2_1_traces_corrections.ipynb)

- [ ] Read the traces and their statuses (4 min)
- [ ] Join the corrections to the traces (5 min)
- [ ] Validate every label against the tool interface (7 min)
- [ ] Write chat-format examples (5 min)
- [ ] Count what was kept and dropped (4 min)
- [ ] ✱ Stretch: Find accepted traces whose call disagrees with the request text

### Session 2: Teacher generation and verifier filtering

#### [day2#s2] Teacher generation and verifier filtering

*Session 2*

- Generate labels with a larger model under frozen instructions
- Filter teacher outputs with schema checks and rule checks
- Measure yield and estimate the errors a verifier lets through

#### [day2#why-a-teacher] A larger model can label requests nobody corrected

*Labels for what the desk never saw*

- The pilot left **900 requests** with no label at all
- On Day 1 the larger model was right on **94 in 100** with the frozen instructions
- Its labels are cheap, fast and in volume, where staff time is none of the three

- **The price.** It is wrong on about 6 in 100, and whatever enters the set is what the student learns. A teacher needs a **verifier**.

#### [day2#teacher-same-format] Same output format as the student, and only data that may leave

*Two conditions before you call the teacher*

- **Format.** The teacher gets the frozen instructions and examples. Its label must be the exact text the student has to produce, or the student learns someone else's format.
- **Placement.** The pool in the lab is synthetic. Real members' requests are Restricted: a hosted teacher may not see them. Run the teacher on your own servers, or label masked and synthetic requests only.

> **The teacher's terms count too.** Its licence and its provider's conditions on training another model with its outputs belong in yesterday's licence table.

#### [day2#verifier-two-layers] A verifier has two layers: the schema, then rules against the request

*What stands between the teacher and the set*

1. **Teacher output** 300
2. **Schema** would the registry accept it? 6 rejected
3. **Rules** does it agree with the request? 1 rejected
4. **Kept** 293

> **A schema knows the interface, not the request.** `"cr_number": "4000000000"` is a valid argument and an invention. Only a rule that reads the request can tell.

#### [day2#rule-check-code] A rule check compares the call with the text it claims to answer

*Lab 2.2 · task 3*

```python
# file: Lab 2.2
numbers = CR.findall(C.western_digits(text))   # numbers the request contains
args = call["arguments"]
if "cr_number" in args and args["cr_number"] not in numbers:
    errors.append("the registration number is not in the request")
if call["name"] == "escalate_to_staff" and args.get("reason") == "missing_cr" and numbers:
    errors.append("escalated as missing although the request gives a number")
if call["name"] == "create_membership" and args.get("company_name", "") not in text:
    errors.append("the company name is not in the request")
```

#### [day2#verifier-blind-spot] Valid, consistent with the request, and wrong

*What passes*

| The teacher's error | Schema | Rules |
| --- | --- | --- |
| Takes the partner's registration number | passes | passes: the number is in the request |
| Renews where a certificate was asked for | passes | passes |
| Writes two years for "one year" | passes | passes |
| Invents a registration number | passes | **caught** |
| Answers in prose | **caught** |  |

#### [day2#yield-is-not-quality] Yield says how much passed, not whether it is right

*Lab 2.2 · tasks 4 and 5*

- **97.7%** yield: 293 of 300 kept
- **8** wrong labels among the 293 kept (answer key): the verifier caught 7 of 15
- **0 of 30** found by the desk's audit, which allows up to 11%

> **For the card:** "audit of 30 kept examples, 0 errors, 95% upper bound 11%", and the kinds of error the verifier cannot see.

#### [day2#check-verifier] What did the verifier prove?

*Checkpoint*

**Question:** A teacher labelled 1,000 requests. Schema and rule checks kept 960. Nobody has read any of them. What can you state about the 960?

- A. They are 96% accurate
- B. They contain no errors, because the bad ones were rejected
- C. They cannot be used until every one is read by staff
- D. They are valid and consistent with their requests; their error rate is unknown until a sample is audited

#### [day2#lab-2-2] Teacher and verifier

Lab 2.2 · 25 min

Label 300 unlabelled requests with the larger model, filter the outputs with schema and rule checks, and measure the yield and the errors that still pass.

[D2_2_teacher_filter.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D2_2_teacher_filter.ipynb)

- [ ] Label 300 requests with the larger model (4 min)
- [ ] Schema checks (5 min)
- [ ] Two rule checks against the request (8 min)
- [ ] Yield, and what was rejected (4 min)
- [ ] Audit 30 kept examples (4 min)
- [ ] ✱ Stretch: Add a third rule and measure whether it rejects a right answer

### Session 3: Chat templates and tool-call formats

#### [day2#s3] Chat templates and tool-call formats

*Session 3*

- Render a conversation with a model's own chat template
- Find formatting faults that harm training: a turn that never ends, a foreign tool-call format, unmasked labels
- Check rendered examples token by token before training

#### [day2#model-sees-one-string] The model never sees your messages: it sees one string

*What the model actually reads*

```text
# file: rendered by the chat template of the course model
<|im_start|>user
نأمل إرسال شهادة العضوية للسجل التجاري 4029184372.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "issue_certificate", "arguments": {"cr_number": "4029184372", "kind": "membership"}}
</tool_call><|im_end|>
```

Role markers, an end-of-turn token, a wrapper for the call. Every one of these is a convention of **this** model, written in its template.

#### [day2#template-is-part-of-model] The chat template is part of the model

*The durable idea*

- The base model was trained on strings in exactly this shape
- Your adapter will be served strings in exactly this shape
- So train on strings in exactly this shape: **use the template, do not rewrite it**

- **What goes wrong.** A hand-written renderer trains "fine": the loss falls, because the model learns whatever text it is given. The fault shows in production, as a format nobody sends.

#### [day2#generation-prompt] The server sends the prompt; training adds the answer

*Two renderings of one example*

- **Prompt.** `add_generation_prompt=True`: the request, then the marker that opens the assistant's turn. The model writes what follows.
- **Full.** The whole conversation, the assistant's turn and its end-of-turn token included. The full text must **start with** the prompt, character for character.

> **That prefix property is what makes the label mask possible:** everything up to the end of the prompt is context, everything after it is the target.

#### [day2#tool-call-formats-differ] Two model families, two ways to write the same call

*Tool-call formats*

##### Qwen3: JSON in tags

```text
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
```

##### LFM3: an XML form in tags

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

Change the base model and the tuning set must be rendered again. Store messages, never rendered strings.

#### [day2#three-faults] Three faults that train "fine" and fail in service

*Lab 2.3*

- **1 · The turn never ends.** No end-of-turn token after the answer. The adapter never learns to stop and writes to the token limit on every request.
- **2 · A foreign call format.** `str(call)`: single quotes, no tags. The model is taught the Python-dictionary failure of Lab 1.1.
- **3 · Loss on everything.** Labels equal to the input. A third of the signal goes into predicting members' requests instead of the answer.

#### [day2#label-mask] The loss is taken on the assistant's turn only

*Fault 3, fixed*

```python
# file: Lab 2.3 · task 4
prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
full_ids   = tok(full,   add_special_tokens=False)["input_ids"]
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
#         ignored by the loss       the tool call and the end of turn
```

> **Why it matters beyond efficiency.** A model trained to predict requests is trained to reproduce what members wrote. Masking the prompt is also a data-protection measure.

#### [day2#round-trip-check] Encode, decode, compare: does the text survive the tokeniser?

*The check people skip*

**written:** الرد على owner289@member.example أو الاتصال على 0509022407.

**decoded (weak example):** الرد على owner289<|unk|>member.example أو الاتصال على 0509022407.

The course tokeniser has no token for the @ sign. Masking removes the address before the tokeniser sees it (Lab 2.4), and the lab's fixed renderer round-trips 50 of 50 masked examples.

#### [day2#check-template] It answers, and then it keeps writing

*Checkpoint*

**Question:** A freshly tuned adapter produces the correct tool call and then continues with more calls and stray text until the token limit. The base model never did this. Where do you look first?

- A. The rendered training text: is there an end-of-turn token after the assistant's answer?
- B. The learning rate: it was too high
- C. The server: raise the stop-token limit
- D. The tuning set: it needs more examples

#### [day2#lab-2-3] Three formatting faults

Lab 2.3 · 25 min

Render examples with the model's chat template, then find and fix three faults in a hand-written renderer that would harm training.

[D2_3_chat_templates.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D2_3_chat_templates.ipynb)

- [ ] Render one example with the model's template (5 min)
- [ ] Fault 1: the turn that never ends (5 min)
- [ ] Fault 2: a tool call the template would not write (6 min)
- [ ] Fault 3: a loss on the whole conversation (6 min)
- [ ] Check the fixed renderer on 50 examples (3 min)
- [ ] ✱ Stretch: Render the same example with a second model's template and list the differences

### Session 4: Deduplication, decontamination and personal data

#### [day2#s4] Deduplication, decontamination and personal data

*Session 4*

- Remove near-duplicates with a stated rule and threshold
- Report overlap with the test set, separating leaks from template collisions
- Scan a tuning set for personal data and mask it

#### [day2#three-silent-problems] Duplicates inflate, leaks flatter, personal data does not belong in weights

*What a loss curve will never show*

- **Near-duplicates.** The same request, sent twice with "Hello" in front. The set looks larger than it is, and one member's wording is over-weighted.
- **Test-set overlap.** A test request in the tuning set. Every score on Day 4 would be too high, by an unknown amount.
- **Personal data.** Names, mobile numbers, identity numbers in free text. Not needed to choose a tool, and a model can reproduce what it was trained on.

#### [day2#near-duplicate-rule] Near-duplicate: one text's three-word runs are contained in the other's

*A rule you can state in one sentence*

**kept:** Kindly extend the membership of CR 4013583134 by 1 year.

**removed:** Kindly extend the membership of CR 4013583134 by 1 year. Regards.

- Normalise: lower case, Western digits, no punctuation
- Compare runs of three words; **containment** = shared runs divided by the smaller set
- Near-duplicate at 0.9 or more, compared only within the same registration number

#### [day2#threshold-tradeoff] At 0.8 the rule merged two different requests

*A threshold is a decision, and it has a cost*

**kept at 0.8:** نرجو إرسال قائمة بجميع الأعضاء العاملين في الاستشارات.

**removed at 0.8 (weak example):** نرجو إرسال قائمة بجميع الأعضاء العاملين في الأجهزة الطبية.

Two short requests that differ only in the activity they ask about: five of the six word runs of the shorter one are shared. At 0.8 they count as one; at 0.9 both stay. The lower threshold removed 82 rows, the higher one 67.

> **Read what a threshold removes** before you accept it. Print ten removed pairs; if one is not a duplicate, the threshold is wrong for your data.

#### [day2#leak-or-collision] A leak is removed; a template collision is kept and counted

*Decontamination*

|  | Leak | Template collision |
| --- | --- | --- |
| What it is | A near-copy of a test request about the same registration number | A short request with no number that many members write alike |
| Example | "We are Al-Salam Supplies Est. from Dammam … CR 4001995900" | "We need a membership certificate as soon as possible." |
| In the lab | 18 removed | 31 kept |
| Why | A test case inside the training data | Removing it would delete a kind of request from training |

#### [day2#personal-data-scan] Patterns find numbers and addresses; names need a list

*Lab 2.4 · tasks 3 and 4*

| Item | Value |
| --- | --- |
| Mobile numbers | 248 |
| Names (from the registry) | 192 |
| Identity numbers | 71 |
| E-mail addresses | 56 |

- Found in 661 rows, then masked: `[PHONE]`, `[NAME]`, `[ID]`, `[EMAIL]`
- A second scan after masking finds 0
- A registration number identifies a company, not a person, and stays

> **"0 found" is a claim about the scan.** A name that is not on the registry's list passes. The card says which kinds and which list.

#### [day2#mask-both-sides] Mask in training and in service, with the same function

*One function, two places*

1. **Request** "… Contact: Fares Al-Sudairy, mobile 0525658054, ID 1168502369."
2. **Mask** the same code in the data pipeline and in front of the server
3. **Model** "… Contact: [NAME], mobile [PHONE], ID [ID]."

> **The model sees in service what it saw in training.** Mask only the tuning set and production sends it digits it never met.

#### [day2#cleaning-changes-mix] Cleaning changed the mix: escalations fell from 19% to 16%

*The stretch of Lab 2.4*

| Item | Value |
| --- | --- |
| Escalations before | 145 of 746 |
| Escalations after | 108 of 661 |

- Requests with no number are short and alike, so deduplication removes more of them
- Escalation is the behaviour Qayd lacks most
- Whether it still learns to escalate is measured on Day 4, on that slice

#### [day2#overlap-report] The overlap report says what was removed, by which rule

*Evidence for outcome 2*

| Step | Rows |
| --- | --- |
| Raw examples (traces, corrections, teacher) | 746 |
| Near-duplicates removed | 67 |
| Leaks from the test set removed | 18 |
| Template collisions kept | 31 |
| Tuning set: 295 traces, 92 corrections, 274 teacher; 336 Arabic, 325 English | 661 |

Rule: containment of three-word runs at 0.9 or more, within the same registration number.

#### [day2#check-overlap] Leak or not?

*Checkpoint*

**Question:** Your overlap check flags a tuning row: "We want to renew for two years. Thank you." The same sentence is in the test set. Neither names a company or a number. What goes in the report?

- A. A leak: remove it, and remove every row like it
- B. A template collision: keep it, and report how many there are
- C. Nothing: generic sentences need no mention
- D. Remove the sentence from the test set instead

#### [day2#lab-2-4] Clean the set

Lab 2.4 · 25 min

Remove near-duplicates from the raw examples, find and remove overlap with the test set, scan for personal data, mask it and write the overlap report.

[D2_4_dedup_pii.ipynb](https://sdaiaacademy.github.io/ae-310/labs/D2_4_dedup_pii.ipynb)

- [ ] Remove near-duplicates (6 min)
- [ ] Find overlap with the test set (7 min)
- [ ] Scan for personal data (5 min)
- [ ] Mask and scan again (4 min)
- [ ] Write the overlap report (3 min)
- [ ] ✱ Stretch: Count escalation examples before and after cleaning: what did cleaning change?

#### [day2#wrap-up] Three things to keep

*Day 2*

- **Every label has a source and a check.** Accepted, corrected, taught by a larger model. Validate each against the tool, and count what you drop.
- **Train on the server's string.** The model's own template, an end of turn, a masked prompt. Compare characters before you compare losses.
- **Clean, and say how.** Duplicates, leaks and personal data, each with a rule a reader can dispute, and the mix after cleaning.

#### [day2#next] Day 3 · Train adapters and distil

*Tomorrow*

The 661 examples meet a model: a low-rank adapter, a 4-bit base, a sweep and a distilled behaviour.

- Guided self-study: open a GPU runtime in Colab once and run the setup check, or decide to train the course model on your laptop
- Recall from AE-301 what a low-rank adapter adds to a layer: tomorrow starts there
