# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1"]
# ///
# %% [markdown]
# # Lab 2.4 · Clean the set
#
# Day 2 · Session 4 (Deduplication, decontamination and personal data) · 25 min.
#
# The raw tuning set is the examples from the traces and from the teacher, 746 rows. Three things are
# wrong with it, and none of them shows in a loss curve: members send the same request twice, some
# requests of the **test set** have found their way into the traces, and the texts carry members' phone
# numbers, identity numbers, e-mail addresses and names. You remove the duplicates, find the overlap,
# mask the personal data and write the overlap report that outcome 2 asks for.
#
# **Reference state.** `data/reference/examples_raw.jsonl` (what Labs 2.1 and 2.2 produce, supplied),
# `data/requests_test.jsonl` and `data/registry.json` (for the owners' names). Nothing of your own from
# an earlier lab is needed.
#
# **Tasks**
# - [ ] 1 · Remove near-duplicates (6')
# - [ ] 2 · Find overlap with the test set (7')
# - [ ] 3 · Scan for personal data (5')
# - [ ] 4 · Mask and scan again (4')
# - [ ] 5 · Write the overlap report (3')
# - [ ] ✱ Stretch · Count escalation examples before and after cleaning: what did cleaning change?

# %%
#@setup
Q.banner("Lab 2.4")

# %%
import copy
import re
from collections import Counter

raw = Q.reference("examples_raw")
test = Q.load("requests_test")
registry = Q.read_json(Q.DATA / "registry.json")
REF = Q.reference("d2_4_clean")
print(len(raw), "raw examples:", dict(Counter(e["source"] for e in raw)), "·", len(test), "test requests")

CR = re.compile(r"(?<![0-9])40[0-9]{8}(?![0-9])")


def text_of(example):
    return example["messages"][0]["content"]


def words(text):
    """Lower case, Western digits, no punctuation."""
    return re.sub(r"[^\w\s]", " ", C.western_digits(text).lower()).split()


def shingles(text):
    """The set of runs of three consecutive words."""
    w = words(text)
    return {" ".join(w[i : i + 3]) for i in range(max(1, len(w) - 2))}


def containment(a, b):
    """The share of the smaller set that the other set contains: 1.0 when one text holds the other."""
    return len(a & b) / max(1, min(len(a), len(b)))


def bucket(text):
    """Only texts about the same registration number are compared; texts with no number share one bucket."""
    numbers = CR.findall(C.western_digits(text))
    return numbers[0] if numbers else "no number"

# %% [markdown]
# ## 1 · Remove near-duplicates (6')
#
# A member who gets no answer sends the request again, with "Hello" in front and "Thank you" behind. An
# exact-match filter misses it. The rule used here: two texts are near-duplicates when the
# **containment** of their three-word shingles is at least 0.9.
#
# ✏️ **Exercise** · Complete `deduplicate`: go through the rows in order and keep a row only if it is
# not a near-duplicate of a row already kept in the same bucket. Return `(kept, removed)`.

# %%
def deduplicate(rows, threshold=0.9):
    kept, removed, seen = [], [], {}  # seen: bucket -> the shingle sets of the rows kept so far
    # BEGIN SOLUTION
    for row in rows:
        s, b = shingles(text_of(row)), bucket(text_of(row))
        if any(containment(s, other) >= threshold for other in seen.get(b, [])):
            removed.append(row)
        else:
            seen.setdefault(b, []).append(s)
            kept.append(row)
    # END SOLUTION
    return kept, removed

# %%
# Check, with a safety net
def _deduplicate(rows, threshold=0.9):
    kept, removed, seen = [], [], {}
    for row in rows:
        s, b = shingles(text_of(row)), bucket(text_of(row))
        if any(containment(s, o) >= threshold for o in seen.get(b, [])):
            removed.append(row)
        else:
            seen.setdefault(b, []).append(s)
            kept.append(row)
    return kept, removed

try:
    deduped, dups = deduplicate(raw)
    if not deduped:
        raise NotImplementedError
except NotImplementedError:
    deduped, dups = _deduplicate(raw)
    print("deduplicate is not written yet: using the reference result")
print(len(dups), "near-duplicates removed,", len(deduped), "rows left")
assert len(dups) == REF["near_duplicates_removed"], len(dups)
d = dups[0]
first = next(r for r in deduped if bucket(text_of(r)) == bucket(text_of(d)) and containment(shingles(text_of(r)), shingles(text_of(d))) >= 0.9)
print("  kept   :", text_of(first)[:110])
print("  removed:", text_of(d)[:110])

# %% [markdown]
# ## 2 · Find overlap with the test set (7')
#
# If a request of the test set, or a near-copy of it, is in the tuning set, the model is examined on
# something it was shown. Every score on Day 4 would be too high, and nobody could say by how much.
#
# Two kinds of overlap look alike and are not:
#
# - a **leak**: a tuning row that nearly copies a test request **about the same registration number**.
#   This is a test case that got into the training data. Remove it.
# - a **template collision**: a short request with no number, such as "We need a membership
#   certificate as soon as possible", which different members write in the same words. It names no
#   member, it will keep arriving in production, and removing it would delete a whole kind of request.
#   Keep it, and report the count.
#
# ✏️ **Exercise** · Complete `overlap`: return `(clean, leaks, collisions)`, where `leaks` are the rows
# removed and `collisions` is the number of rows kept although their text overlaps a test request.

# %%
test_seen = {}
for t in test:
    test_seen.setdefault(bucket(t["text"]), []).append(shingles(t["text"]))


def overlaps_test(row, threshold=0.9):
    s, b = shingles(text_of(row)), bucket(text_of(row))
    return any(containment(s, other) >= threshold for other in test_seen.get(b, []))


def overlap(rows):
    clean, leaks, collisions = [], [], 0
    # BEGIN SOLUTION
    for row in rows:
        hit = overlaps_test(row)
        if hit and bucket(text_of(row)) != "no number":
            leaks.append(row)
            continue
        collisions += hit
        clean.append(row)
    # END SOLUTION
    return clean, leaks, collisions

# %%
try:
    clean, leaks, collisions = overlap(deduped)
    if not clean:
        raise NotImplementedError
except NotImplementedError:
    clean = [r for r in deduped if not (overlaps_test(r) and bucket(text_of(r)) != "no number")]
    leaks = [r for r in deduped if overlaps_test(r) and bucket(text_of(r)) != "no number"]
    collisions = sum(overlaps_test(r) for r in clean)
    print("overlap is not written yet: using the reference result")
print(len(leaks), "leaks removed ·", collisions, "template collisions kept ·", len(clean), "rows left")
assert len(leaks) == REF["leaks_removed"] and collisions == REF["template_collisions_kept"], (len(leaks), collisions)
print("a leak     :", text_of(leaks[0])[:110])
print("a collision:", next(text_of(r) for r in clean if overlaps_test(r))[:110])

# %% [markdown]
# ## 3 · Scan for personal data (5')
#
# A request ends with "Contact: …, mobile 05…, ID 1…". None of that is needed to choose a tool, and a
# model can reproduce what it was trained on. `C.find_personal_data(text, names)` returns what it finds,
# by kind: mobile numbers and identity numbers by pattern, e-mail addresses by pattern, and names from
# a list, here the owners' names in the registry. A registration number identifies a company, not a
# person, and is kept.
#
# ✏️ **Exercise** · Complete `scan`: return a `Counter` of personal data found in the rows, by kind.

# %%
names = sorted({r[k] for r in registry for k in ("owner_en", "owner_ar")})


def scan(rows):
    found = Counter()
    # BEGIN SOLUTION
    for row in rows:
        for kind, values in C.find_personal_data(text_of(row), names).items():
            found[kind] += len(values)
    # END SOLUTION
    return found

# %%
try:
    found = scan(clean)
    if not found:
        raise NotImplementedError
except NotImplementedError:
    found = Counter(REF["personal_data_found"])
    print("scan is not written yet: using the supplied counts")
print("personal data in", len(clean), "rows:", dict(found))
assert dict(found) == REF["personal_data_found"], dict(found)

# %% [markdown]
# ## 4 · Mask and scan again (4')
#
# `C.mask_personal_data` replaces each find with a placeholder: `[PHONE]`, `[ID]`, `[EMAIL]`, `[NAME]`.
# The same function runs in front of the model in service, so the model always sees what it was trained
# on. Then scan again: a masking step that is not checked is a hope.

# %%
masked = copy.deepcopy(clean)
for row in masked:
    row["messages"][0]["content"] = C.mask_personal_data(text_of(row), names)
left = sum(len(v) for row in masked for v in C.find_personal_data(text_of(row), names).values())
print("before:", text_of(next(r for r in clean if "ID" in text_of(r) or "هوية" in text_of(r))))
print("after :", text_of(next(r for r in masked if "[ID]" in text_of(r))))
print("personal data left after masking:", left)
assert left == 0 and len(masked) == REF["rows_out"]
assert masked == Q.load("tuning/train"), "the cleaned set differs from the supplied data/tuning/train.jsonl"

# %% [markdown]
# 👀 **Look at** what the scan can and cannot promise. Numbers and addresses are found by pattern. Names
# are found only because the registry lists them: a name that is not on the list (a new contact person,
# a misspelling) passes. The dataset card must say "names masked from the registry's list of owners",
# not "no personal data".

# %% [markdown]
# ## 5 · Write the overlap report (3')

# %%
report = {
    "rows_in": len(raw), "near_duplicates_removed": len(dups), "leaks_removed": len(leaks), "template_collisions_kept": collisions,
    "personal_data_masked": dict(found), "personal_data_left": left, "rows_out": len(masked),
    "by_source": dict(Counter(r["source"] for r in masked)), "by_lang": dict(Counter(r["lang"] for r in masked)),
    "rule": "containment of three-word shingles at 0.9 or more, compared within the same registration number",
}
Q.show("Overlap report · tuning set against the test set", [(k, str(v)) for k, v in report.items()])
out = Q.artifacts("2.4")
Q.write_json(out / "overlap_report.json", report)
Q.write_jsonl(out / "train.jsonl", masked)
print("saved", (out / "overlap_report.json").relative_to(Q.LABS))

# %% [markdown]
# ## ✱ Stretch · What cleaning changed

# %%
def escalations(rows):
    return sum(r["messages"][1]["tool_calls"][0]["function"]["name"] == "escalate_to_staff" for r in rows)


print(f"escalation examples: {escalations(raw)} of {len(raw)} before ({escalations(raw) / len(raw):.0%}), {escalations(masked)} of {len(masked)} after ({escalations(masked) / len(masked):.0%})")
print("Requests with no registration number are short and alike, so deduplication removes more of them than of anything else.")
print("Cleaning changed the mix of the set. Whether the model still learns to escalate is a question for Day 4, by slice.")

# %% [markdown]
# ## Check yourself
#
# 1. Why are template collisions kept, although their text is in the test set?
# 2. The leaks were found among the traces. How did test requests get there?
# 3. What is the weakest claim in "personal data left after masking: 0"?
#
# <details><summary>Answers</summary>
#
# 1. They carry no member's content and are how real members write. Removing them would delete a kind of
#    request from training; the report states how many there are, so a reader can judge.
# 2. The test set was drawn from the same service as the pilot. A member who wrote during the test
#    period wrote again during the pilot. Splitting by company prevents it only if the split is applied
#    to every source.
# 3. "Zero found by this scan." The scan knows four kinds and one list of names.
#
# </details>
#
# **Carry forward.** `artifacts/2.4/train.jsonl` and `overlap_report.json`. The supplied copy,
# `data/tuning/train.jsonl`, is what every Day 3 lab trains on.
