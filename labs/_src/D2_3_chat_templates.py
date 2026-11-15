# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.80", "python-dotenv>=1.1", "torch>=2.6", "transformers>=4.56", "jinja2>=3.1"]
# ///
# %% [markdown]
# # Lab 2.3 · Three formatting faults
#
# Day 2 · Session 3 (Chat templates and tool-call formats) · 25 min.
#
# A model never sees your list of messages. It sees one string, built by its **chat template**: role
# markers, special tokens, and a precise way of writing a tool call. Train on a string that differs from
# the one the server will build, and the adapter learns a format that production never sends. A
# colleague wrote a renderer by hand, with three faults that would each harm training. You render with
# the model's own template, find the faults and fix them.
#
# **Reference state.** `data/tuning/train.jsonl` (the supplied, cleaned tuning set) and the
# tokeniser of the course model in `models/qayd-tiny/`, which carries a chat template like any model on
# the Hub. No model is loaded and nothing is trained.
#
# **Tasks**
# - [ ] 1 · Render one example with the model's template (5')
# - [ ] 2 · Fault 1: the turn that never ends (5')
# - [ ] 3 · Fault 2: a tool call the template would not write (6')
# - [ ] 4 · Fault 3: a loss on the whole conversation (6')
# - [ ] 5 · Check the fixed renderer on 50 examples (3')
# - [ ] ✱ Stretch · Render the same example with a second model's template and list the differences

# %%
#@setup
Q.banner("Lab 2.3")

# %%
import tuning as U

tok = U.load_tokenizer("tiny")
examples = Q.load("tuning/train")[:50]
example = examples[2]
print("special tokens:", tok.pad_token, tok.eos_token)
print()
print("The chat template of the model (Jinja):")
print(tok.chat_template)

# %% [markdown]
# ## 1 · Render one example with the model's template (5')
#
# `tok.apply_chat_template(messages, tokenize=False)` returns the exact string. With
# `add_generation_prompt=True` it stops after the marker that opens the assistant's turn: that is what
# the server sends, and the model writes the rest.
#
# ✏️ **Exercise** · Complete `render`: return `(prompt, full)` for an example. `prompt` is what the
# server sends (the user turn and the opened assistant turn); `full` is the whole conversation, which
# training uses.

# %%
def render(example):
    messages = example["messages"]
    # BEGIN SOLUTION
    prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    return prompt, full
    # END SOLUTION

# %%
# Check, with a safety net
def _render(example):
    m = example["messages"]
    return tok.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True), tok.apply_chat_template(m, tokenize=False)

try:
    prompt, full = render(example)
    assert (prompt, full) == _render(example)
except NotImplementedError:
    render = _render
    prompt, full = render(example)
    print("render is not written yet: using the reference version")
assert full.startswith(prompt), "the full text continues the prompt exactly: that is what makes the label mask of task 4 possible"
print("What the server sends:")
print(prompt)
print("-" * 60)
print("What training adds (the assistant's turn):")
print(full[len(prompt):])

# %% [markdown]
# Now the colleague's renderer. It looks reasonable, and it produced a file that "trained fine".

# %%
def naive_render(example):
    """A hand-written renderer with three faults. Do not fix it here: find the faults first."""
    user = example["messages"][0]["content"]
    call = example["messages"][1]["tool_calls"][0]["function"]
    text = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + str(call)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text, ids, list(ids)  # the text, its token ids, and the labels the loss is taken on


naive_text, naive_ids, naive_labels = naive_render(example)
print(naive_text)

# %% [markdown]
# ## 2 · Fault 1: the turn that never ends (5')
#
# A model stops because it was trained to write an end-of-turn token. If the training text does not
# contain it after the assistant's answer, the adapter never learns to stop: in service it answers, then
# keeps writing until the token limit, and every request costs the maximum.
#
# ✏️ **Exercise** · Complete `ends_turn`: does a rendered training text close the assistant's turn with
# the model's end-of-turn token?

# %%
def ends_turn(text):
    # BEGIN SOLUTION
    return text.rstrip("\n").endswith(tok.eos_token)
    # END SOLUTION

# %%
try:
    assert ends_turn(full) and not ends_turn(naive_text)
except NotImplementedError:
    def ends_turn(text):
        return text.rstrip("\n").endswith(tok.eos_token)
    print("ends_turn is not written yet: using the reference version")
print("template text closes the turn:", ends_turn(full), "· hand-written text closes the turn:", ends_turn(naive_text))

# %% [markdown]
# ## 3 · Fault 2: a tool call the template would not write (6')
#
# The template writes a tool call in one precise way. The server's parser expects exactly that, and so
# does `C.parse_call`. Compare the two strings and find where they part.
#
# ✏️ **Exercise** · Complete `first_difference`: return the index of the first character at which two
# strings differ (or `None` when they are equal).

# %%
def first_difference(a, b):
    # BEGIN SOLUTION
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
    # END SOLUTION

# %%
try:
    assert first_difference("abc", "abd") == 2 and first_difference("ab", "ab") is None and first_difference("ab", "abc") == 2
    at = first_difference(full, naive_text)
except NotImplementedError:
    at = next(i for i, (x, y) in enumerate(zip(full, naive_text)) if x != y)
    print("first_difference is not written yet: using the reference result")
print("the two texts part at character", at)
print("  template    :", repr(full[at - 12 : at + 60]))
print("  hand-written:", repr(naive_text[at - 12 : at + 60]))
print()
print("Can the registry's parser read the hand-written call?", C.parse_call(naive_text[len(prompt):]) is not None)
print("Can it read the template's?                          ", C.parse_call(full[len(prompt):]) is not None)

# %% [markdown]
# 👀 **Look at** the difference. `str(call)` is Python's way of printing a dictionary: single quotes, no
# tags. It is not JSON and it is not what the template writes. An adapter trained on it would produce
# exactly the `python_dict` format failure you counted in Lab 1.1, and it would have been taught to.

# %% [markdown]
# ## 4 · Fault 3: a loss on the whole conversation (6')
#
# The hand-written renderer returns `labels = ids`: the loss is taken on every token, the member's
# request included. Most of the training signal then goes into predicting what members write, which is
# not the task, and in a real project means the adapter is trained to reproduce members' text.
#
# The fix is a **label mask**: `-100` on every token of the prompt (PyTorch's loss ignores it), the real
# token ids on the assistant's turn.
#
# ✏️ **Exercise** · Complete `masked_labels`: given the token ids of the prompt and of the full text,
# return the labels.

# %%
def masked_labels(prompt_ids, full_ids):
    # BEGIN SOLUTION
    return [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    # END SOLUTION

# %%
prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
full_ids = tok(full, add_special_tokens=False)["input_ids"]
assert full_ids[: len(prompt_ids)] == prompt_ids
try:
    labels = masked_labels(prompt_ids, full_ids)
    assert len(labels) == len(full_ids) and labels[: len(prompt_ids)] == [-100] * len(prompt_ids) and labels[-1] == full_ids[-1]
except NotImplementedError:
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    print("masked_labels is not written yet: using the reference result")
supervised = sum(l != -100 for l in labels)
print(f"tokens in the example: {len(full_ids)} · supervised with the mask: {supervised} · supervised by the hand-written renderer: {len(naive_labels)}")
print(f"Without the mask, {1 - supervised / len(full_ids):.0%} of the loss is spent on the request, not the answer.")
print("The supervised part, decoded:", repr(tok.decode([l for l in labels if l != -100])))

# %% [markdown]
# ## 5 · Check the fixed renderer on 50 examples (3')
#
# The fixed renderer is short, because the fix is to stop writing templates by hand.

# %%
def fixed_render(example):
    prompt, full = render(example)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"]
    return full, f, [-100] * len(p) + f[len(p):]


report = {"closes the turn": 0, "call parses and matches the label": 0, "labels masked": 0, "decodes back to the text": 0}
shares = []
for e in examples:
    text, ids, labs = fixed_render(e)
    p, _ = render(e)
    report["closes the turn"] += ends_turn(text)
    report["call parses and matches the label"] += C.same_call(C.parse_call(text[len(p):]), e["messages"][1]["tool_calls"][0]["function"])
    report["labels masked"] += labs[0] == -100 and labs[-1] != -100
    report["decodes back to the text"] += tok.decode(ids) == text
    shares.append(sum(l != -100 for l in labs) / len(labs))
for k, v in report.items():
    print(f"  {k}: {v} of {len(examples)}")
print(f"  share of tokens supervised: {sum(shares) / len(shares):.0%} on average")
assert all(v == len(examples) for v in report.values()), report
Q.write_json(Q.artifacts("2.3") / "render_report.json", report)

# %% [markdown]
# 👀 **Look at** the fourth line. "Decodes back to the text" is the check people skip: encode, decode,
# compare. The cell below shows it failing: a raw request from the pilot, before masking, holds an
# e-mail address, and this tokeniser has no token for the `@` sign.

# %%
raw_text = next(t["text"] for t in Q.load("traces_500") if "@" in t["text"])
back = tok.decode(tok(raw_text, add_special_tokens=False)["input_ids"])
print("written:", raw_text[-70:])
print("decoded:", back[-70:])
print("round trip holds:", back == raw_text)

# %% [markdown]
# The unknown-token marker is what the model would have been trained on. Here masking removes the
# address before the tokeniser sees it (Lab 2.4), so the tuning set is safe. With your own data and your
# own model, run this check on every example before the first training step.

# %% [markdown]
# ## ✱ Stretch · A second model's template
#
# The hub tier of Day 3 uses a real open-weight model. Its tokeniser files (a few megabytes, no weights)
# carry its template. Render the same example with it, pass the tool definitions, and list what differs:
# where the tools are described, how a call is wrapped, whether there is a reasoning switch.

# %%
try:
    hub = U.load_tokenizer("hub")
    text = hub.apply_chat_template([{"role": "system", "content": "You are Qayd."}] + example["messages"], tools=C.TOOLS[:1], tokenize=False, enable_thinking=False)
    print(U.HUB_BASE, "renders the example as:")
    print(text[-420:])
except Exception as error:  # no network, or the Hub did not answer
    print("The second tokeniser could not be fetched here:", type(error).__name__)
    print("Try again in Colab, or read the chat template on the model's page.")

# %% [markdown]
# ## Check yourself
#
# 1. An adapter answers correctly and then keeps generating text until the token limit. Which fault?
# 2. Why is "it trained fine, the loss went down" no evidence that the rendering was right?
# 3. Where must the same template be used besides training?
#
# <details><summary>Answers</summary>
#
# 1. The training text had no end-of-turn token after the answer: the model was never shown stopping.
# 2. The loss measures how well the model predicts the text you gave it. If the text is wrong, a low
#    loss means it learnt the wrong text well.
# 3. In evaluation and in the server. A server that builds the prompt with a different template (or a
#    different version of it) sends the adapter a string it was not trained on.
#
# </details>
#
# **Carry forward.** The renderer of Days 3 and 4, `tuning.encode`, does what `fixed_render` does.
