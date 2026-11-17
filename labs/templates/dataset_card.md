# Dataset card · Qayd registration tuning set

**What it is.** {rows} tuning examples for one task: a member's registration request, in Arabic or
English, answered by exactly one call to the registry's tool interface. All requests are synthetic,
written for the course; no value belongs to a person or a company.

**Where the examples come from.**

| Source | Examples | Label written by |
|---|---|---|
| Accepted pilot traces | {from_traces} | the pilot model; nobody objected |
| Staff corrections | {from_corrections} | the registry desk |
| Teacher model, filtered | {from_teacher} | a larger model, checked by schema and rules |

Languages: {arabic} Arabic, {english} English.

**What was left out, and by which rule.**

- {abandoned} abandoned traces: no label exists.
- {corrections_rejected} staff corrections: the tool interface rejects the call as typed.
- Teacher labels that failed the schema or a rule; yield {teacher_yield}.
- {near_duplicates} near-duplicates: containment of three-word runs at 0.9 or more, within the same registration number.
- {leaks} leaks from the test set: a near-copy of a test request about the same registration number.

**What was kept and counted.** {collisions} template collisions: short requests with no registration
number whose text also occurs in the test set.

**Personal data.** Found and masked before training: {personal_data}. Names are found from the
registry's list of owners only. A second scan after masking found 0. The same masking function runs
in front of the model in service.

**Known limits.**

- Accepted traces were not reviewed: some carry a wrong call that passes every check.
- Teacher labels: an audit of {audit_n} kept examples found {audit_wrong} errors, which allows a true error rate of up to {audit_bound}. Errors that are valid and consistent with the request are not caught by the verifier.
- Abandoned traces were the hardest requests: the set is easier than production.
- Cleaning lowered the share of escalation examples.

**Use.** Tuning and evaluating models for this task inside the course. Not a source of facts about any
real chamber of commerce.
