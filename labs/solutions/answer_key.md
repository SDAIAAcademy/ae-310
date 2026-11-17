# Answer key · the planted problems

For trainers. Every problem below was put into the case on purpose by `prepare_data.py` (seed 310), `_tools/make_reference.py` and `_tools/train_reference.py`. The counts are those of the course's own run: Days 1 and 2 with the scripted stand-in for the hosted models, Days 3 and 4 with the course model on a processor. With a real key, or on the hub tier, the counts differ and the labs skip the assertions that depend on them.

## Day 1

| Lab | What is planted | Where it shows | Count |
|---|---|---|---|
| 1.1 | Three failure classes in the small model's outputs: knowledge (a wrong or invented activity code), format (prose, a Markdown fence, a Python dictionary, Arabic-Indic digits in the call), behaviour (the partner's number, a missed escalation, the wrong tool, the wrong number of years) | `data/reference/outputs/small_v0_100.jsonl` | small model 27 of 100 right; failures: knowledge 10, format 37, behaviour 26. Larger model 63 right; 23, 9, 5 |
| 1.2 | A weak first prompt, so that an unfair baseline can be shown | `prompts.py`, versions 0 to 4 | 27, 35, 50, 52, 67 of 100; frozen baseline 67 (95% interval 57% to 75%); the larger model with the same prompt 94 |
| 1.3 | Three licences that answer the chamber's three uses differently: Apache-2.0 (all three allowed), a revenue-capped licence (the chamber's revenue is above the cap), a non-commercial licence (the registration service charges fees) | `data/licences/`, `data/chamber_facts.json` | one model of three passes all three uses |
| 1.4 | The hosted model is far cheaper per request than the chamber's own machine at the expected volume; placement, not cost, decides | `data/prices.json`, `data/onprem.json` | hosted SAR 0.52 per 1,000; own machine SAR 5,655 a month, SAR 141.38 per 1,000 at 40,000 requests a month; break-even near 11 million requests a month |

## Day 2

| Lab | What is planted | Where it shows | Count |
|---|---|---|---|
| 2.1 | Abandoned traces with no label | `status == "abandoned"` | 35 of 500 |
| 2.1 | Corrections the tool interface rejects: a number one digit short, a city in Arabic, Arabic-Indic digits | `data/corrections.jsonl` | 12 of 123 |
| 2.1 | Accepted traces whose call is wrong | answer key in `data/reference/d2_1_examples.json` (`wrong_labels_left`) | 8 of 342; the stretch rule finds 2 (R-0058, R-0179) |
| 2.2 | Teacher errors of two kinds: detectable (prose, an invented number, a wrong escalation) and undetectable (the partner's number, the wrong tool, the wrong years) | `data/reference/outputs/teacher_pool.jsonl` | 15 wrong of 300; schema rejects 6, rules 1; 8 wrong labels kept among 293 (2.7%) |
| 2.2 | An audit that finds nothing | `data/reference/audit_30.jsonl` | 0 of 30; 95% upper bound 11.4% |
| 2.3 | A hand-written renderer with three faults: no end-of-turn token, the call printed as a Python dictionary, labels on the whole text | `naive_render` in the notebook | the example has 82 tokens, 56 supervised with the mask |
| 2.3 | A tokeniser without the `@` sign | the round-trip cell | an unmasked e-mail address decodes with an unknown-token marker |
| 2.4 | Near-duplicates: the same request sent again with a greeting | `data/reference/examples_raw.jsonl` | 67 removed of 746 at threshold 0.9 |
| 2.4 | Test requests that reappear in the pilot traces | 23 planted in the 500 traces | 18 reach this step with a registration number and are removed |
| 2.4 | Template collisions: short requests with no number that also occur in the test set | | 31 kept and reported |
| 2.4 | Personal data in free text | | 248 mobile numbers, 192 names, 71 identity numbers, 56 e-mail addresses; 0 after masking |

## Days 3 and 4

The numbers of Days 3 and 4 are in the supplied files and are printed by the solution notebooks; they are not repeated here, so that this key cannot drift from them.

| Lab | What is planted | Supplied file |
|---|---|---|
| 3.1 | A run with a learning rate fifty times too high, no warm-up and no decay | `data/reference/d3_1_bad_run.json` |
| 3.3 | One missing cell in the sweep (rank 32 at learning rate 0.002) | `data/reference/d3_3_sweep.json`, key `held_out` |
| 3.4 | Wrong teacher labels that the verifier kept, with their right calls | `data/reference/d3_4_teacher_errors.jsonl` |
| 4.2 | An adapter trained with no example of refusing or of the general tasks, and one trained with 200 replay examples | `adapters/registry-4bit/`, `adapters/registry-replay/`, `data/reference/d4_checks.json` |

## Not planted

Two findings came out of building the course and were kept because they teach: the first deduplication threshold (0.8) merged different short requests (Day 2, session 4), and the course tokeniser cannot represent the `@` sign (Day 2, session 3).
