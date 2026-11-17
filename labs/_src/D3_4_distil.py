# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.6", "transformers>=4.56", "peft>=0.17", "jinja2>=3.1"]
# ///
# %% [markdown]
# # Lab 3.4 · Distil a behaviour
#
# Day 3 · Session 4 (Distilling a behaviour) · 25 min.
#
# Two ways to teach Qayd the decisions it gets wrong. The registry desk's corrections: few, written by
# people who know the registry. The larger model's filtered tool calls: many, cheap, and wrong now and
# then in ways no verifier saw (Lab 2.2). You train a student on the teacher's calls with a short
# schedule and compare it, slice by slice, with a supplied adapter trained on corrections alone.
#
# **Reference state.** `data/tuning/teacher_filtered.jsonl` (the verifier's output over the whole
# pool), `data/tuning/corrections_only.jsonl`, the supplied adapter `adapters/corrections-only/`, and
# `data/reference/d3_4_teacher_errors.jsonl`: the answer key for the teacher's kept labels, which no
# real project has.
#
# **Tasks**
# - [ ] 1 · Compare the two tuning sets (4')
# - [ ] 2 · Train the student on the teacher's filtered calls (9')
# - [ ] 3 · Score both adapters on the behaviour slices (7')
# - [ ] 4 · Find a teacher error the student copied (5')
# - [ ] ✱ Stretch · Train on corrections and teacher calls together and compare

# %%
#@setup
Q.banner("Lab 3.4")

# %%
from collections import Counter

import tuning as U

TIER = U.tier()
teacher_rows = U.rows_from(Q.load("tuning/teacher_filtered"))
correction_rows = U.rows_from(Q.load("tuning/corrections_only"))
val_rows = U.rows_from(Q.load("tuning/val"))
val = Q.load("requests_val")
if Q.QUICK:
    val = val[:32]
texts = [r["text"] for r in val]

# %% [markdown]
# ## 1 · Compare the two tuning sets (4')
#
# `U.slices(row)` names the behaviour slices a request belongs to, from its text and its call:
# `escalation` (the right answer is to hand over to staff), `two numbers` (the request names two
# registration numbers), `Arabic-Indic digits`, or `plain`.
#
# ✏️ **Exercise** · Complete `slice_counts`: how many rows of a tuning set fall in each slice?

# %%
def slice_counts(rows):
    counts = Counter()
    # BEGIN SOLUTION
    for row in rows:
        for name in U.slices({"text": row["text"], "gold": row["call"]}):
            counts[name] += 1
    # END SOLUTION
    return counts

# %%
# Check, with a safety net
def _slice_counts(rows):
    return Counter(name for row in rows for name in U.slices({"text": row["text"], "gold": row["call"]}))

try:
    counts = {"corrections": slice_counts(correction_rows), "teacher": slice_counts(teacher_rows)}
    if not counts["teacher"]:
        raise NotImplementedError
    assert counts["teacher"] == _slice_counts(teacher_rows)
except NotImplementedError:
    counts = {"corrections": _slice_counts(correction_rows), "teacher": _slice_counts(teacher_rows)}
    print("slice_counts is not written yet: using the reference counts")
names = ["plain", "escalation", "two numbers", "Arabic-Indic digits"]
print(f"{'':<22}{'corrections':>12}{'teacher':>10}")
print(f"{'examples':<22}{len(correction_rows):>12}{len(teacher_rows):>10}")
for n in names:
    print(f"{n:<22}{counts['corrections'][n]:>12}{counts['teacher'][n]:>10}")

# %% [markdown]
# 👀 **Look at** the mix. The corrections are concentrated where Qayd failed, because that is why the
# desk corrected them. The teacher's set has the mix of the pool, which is the mix of real requests.

# %% [markdown]
# ## 2 · Train the student on the teacher's filtered calls (9')
#
# A short schedule: 200 steps. The student is not learning a new task from nothing; it is learning one
# behaviour from many consistent examples.
#
# ✏️ **Exercise** · Complete `distil`: add a rank-8 adapter to the base and train it on the teacher's
# rows for `STEPS` steps (learning rate `2e-3`, batch 8). Return `(model, log)`.

# %%
STEPS = Q.pick(200, 30)
base, tok = U.load_base()


def distil(base):
    # BEGIN SOLUTION
    model = U.add_adapter(base, rank=8)
    log = U.train(model, tok, teacher_rows, val_rows, steps=STEPS, lr=2e-3, batch=8)
    return model, log
    # END SOLUTION

# %%
try:
    student, log = distil(base)
except NotImplementedError:
    student = U.add_adapter(base, rank=8)
    log = U.train(student, tok, teacher_rows, val_rows, steps=STEPS, lr=2e-3, batch=8)
    print("distil is not written yet: ran with the course's settings")
print(f"{log['steps']} steps in {log['seconds']:.0f} s · validation loss {log['curve'][0]['val_loss']} → {log['curve'][-1]['val_loss']}")

# %% [markdown]
# ## 3 · Score both adapters on the behaviour slices (7')
#
# ✏️ **Exercise** · Complete `by_slice`: from graded outputs and their requests, return
# `{slice: (right, total)}`.

# %%
def by_slice(graded, rows):
    table = {}
    # BEGIN SOLUTION
    for g, row in zip(graded, rows):
        for name in U.slices(row):
            right, total = table.get(name, (0, 0))
            table[name] = (right + g["correct"], total + 1)
    # END SOLUTION
    return table

# %%
def _by_slice(graded, rows):
    table = {}
    for g, row in zip(graded, rows):
        for name in U.slices(row):
            right, total = table.get(name, (0, 0))
            table[name] = (right + g["correct"], total + 1)
    return table

student_graded = U.grade(U.generate(student, tok, texts), val)
base2, _ = U.load_base()
if TIER == "tiny":
    corrections = U.load_adapter(base2, Q.ADAPTERS / "corrections-only")
    corrections_graded = U.grade(U.generate(corrections, tok, texts), val)
else:  # no adapter is supplied for the hub model: train one on the corrections here
    corrections = U.add_adapter(base2, rank=8)
    U.train(corrections, tok, correction_rows, val_rows, steps=Q.pick(150, 30), lr=2e-3, batch=8)
    corrections_graded = U.grade(U.generate(corrections, tok, texts), val)
try:
    s_table, c_table = by_slice(student_graded, val), by_slice(corrections_graded, val)
    if not s_table:
        raise NotImplementedError
except NotImplementedError:
    s_table, c_table = _by_slice(student_graded, val), _by_slice(corrections_graded, val)
    print("by_slice is not written yet: using the reference tables")
print(f"{'slice':<22}{'corrections only':>18}{'distilled':>14}")
for n in names:
    if n in s_table:
        print(f"{n:<22}{c_table[n][0]:>12} of {c_table[n][1]:<3}{s_table[n][0]:>8} of {s_table[n][1]:<3}")
total_c, total_s = sum(g["correct"] for g in corrections_graded), sum(g["correct"] for g in student_graded)
print(f"{'all':<22}{total_c:>12} of {len(val):<3}{total_s:>8} of {len(val):<3}")

# %% [markdown]
# 👀 **Look at** the slices with fewer than twenty requests before you read them. A slice of nine
# requests moves by eleven points for each request. The table says where to look, not how large the
# difference is.

# %% [markdown]
# ## 4 · Find a teacher error the student copied (5')
#
# The verifier let through labels that are valid, consistent with their request, and wrong. The answer
# key lists them. Give those same requests to the student: where its answer equals the teacher's wrong
# label, it has learnt the error.

# %%
wrong = Q.reference("d3_4_teacher_errors")
outs = U.generate(student, tok, [w["text"] for w in wrong])
copied = [(w, o) for w, o in zip(wrong, outs) if C.same_call(C.parse_call(o), w["teacher"])]
fixed = [(w, o) for w, o in zip(wrong, outs) if C.same_call(C.parse_call(o), w["gold"])]
print(f"{len(wrong)} wrong labels among the {len(teacher_rows)} the verifier kept ({len(wrong) / len(teacher_rows):.1%})")
print(f"on those requests the student gives the teacher's wrong call {len(copied)} times and the right call {len(fixed)} times")
for w, o in copied[:2]:
    print()
    print("request :", w["text"][:150])
    print("teacher :", w["teacher"])
    print("right   :", w["gold"])
Q.write_json(Q.artifacts("3.4") / "distil.json", {"steps": log["steps"], "val_right": total_s, "corrections_only_right": total_c, "of": len(val),
                                                  "teacher_errors": len(wrong), "copied": len(copied), "answered_right": len(fixed)})
U.save_adapter(student, Q.artifacts("3.4") / "adapter", log)

# %% [markdown]
# 👀 **Look at** the two counts. A request the student answers with the teacher's wrong call is a
# training example it reproduced. A request it answers correctly although its label was wrong shows the
# other side: a few wrong labels among hundreds of consistent ones are outvoted. Which of the two
# happens depends on how systematic the teacher's error is. An error the teacher makes every time on one
# kind of request is not noise to the student: it is the lesson.

# %% [markdown]
# ## ✱ Stretch · Both sources together

# %%
if Q.QUICK:
    print("quick mode: the stretch is skipped")
else:
    base3, _ = U.load_base()
    both = U.add_adapter(base3, rank=8)
    U.train(both, tok, teacher_rows + correction_rows, val_rows, steps=STEPS, lr=2e-3, batch=8)
    both_graded = U.grade(U.generate(both, tok, texts), val)
    print(f"teacher and corrections together: {sum(g['correct'] for g in both_graded)} of {len(val)} · teacher alone: {total_s} · corrections alone: {total_c}")

# %% [markdown]
# ## Check yourself
#
# 1. The corrections-only adapter was trained on 92 examples. What is its main risk?
# 2. The student copies an error of the teacher. Where in the pipeline do you fix it?
# 3. When is distillation the wrong tool?
#
# <details><summary>Answers</summary>
#
# 1. It has seen almost only the hard cases: it can lose the plain ones, and with so few examples it
#    memorises. Its score by slice shows which.
# 2. In the data: a rule in the verifier if the error can be stated, a larger audit if it cannot, or
#    corrections for exactly that kind of request. Not in the training settings.
# 3. When the teacher is not clearly better than the student on the behaviour, when its outputs may not
#    be used for training under its terms, or when the requests may not be sent to it.
#
# </details>
#
# **Carry forward.** `artifacts/3.4/adapter/` and `distil.json`. The supplied copy is `adapters/distilled/`.
