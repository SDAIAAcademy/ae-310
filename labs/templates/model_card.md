# Model card · Qayd registry adapter

**What it is.** A low-rank adapter for {base_name}. {precision}.

**Intended use.** Turning one registration request, in Arabic or English, into one call to the
registry's tool interface. **Not for** general questions, advice, or any request for a member's
personal data.

**Training.** {train_rows} examples (see the dataset card), {steps} steps, learning rate {lr}, batch
{batch}, seed {seed}; {trainable} trainable parameters; {seconds} s on a processor. The run log with
both loss curves is stored beside the adapter.

**Evaluation on the held-out test set ({of} requests, exact match of the tool call).**

| System | Right |
|---|---|
| Base model, no adapter | {base} |
| This adapter | {tuned} |
| Larger prompted model (reference) | {large} |

- Adapter against base: {vs_base}
- Adapter against the larger prompted model: {vs_large}
- Arabic: {ar_tuned} of {ar_n}; against the larger model {ar_vs_large}
- English: {en_tuned} of {en_n}; against the larger model {en_vs_large}

Intervals are 95% paired bootstrap intervals over the test requests. One training run, one seed.

**What got worse.**

| Check | Base | This adapter | Adapter trained with replay |
|---|---|---|---|
| General abilities, of {general_of} | {general_base} | {general_tuned} | {general_replay} |
| Requests for personal data refused, of {refusal_of} | {refusal_base} | {refusal_tuned} | {refusal_replay} |

The replay adapter scores {test_replay} of {of} on the test set. The checks cover five general tasks
and one kind of refusal; abilities outside them are unmeasured.

**Limits.** Evaluated on synthetic requests from the same generator as the tuning set. Requests unlike
these (other services, other languages, long letters) are untested.
