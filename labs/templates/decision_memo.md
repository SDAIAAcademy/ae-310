# Decision memo · Qayd tuned model

To: Maysoon Al-Dabbagh · From: the adaptation team · One page

## Decision: {decision}

{conditions}

## Quality

On {of} held-out requests the tuned model answers {tuned} correctly, its base {base}, and the larger
prompted model {large}. Tuned against base: {vs_base}. Tuned against the larger model: {vs_large}.
Arabic {ar_tuned} of {ar_n}, English {en_tuned} of {en_n}.

## What got worse

General abilities: {general_base} before, {general_tuned} after, of {general_of}. Requests for
personal data refused: {refusal_base} before, {refusal_tuned} after, of {refusal_of}. With replay:
{general_replay} and {refusal_replay}, at {test_replay} of {of} on the test set.

## Cost

Hosted larger model: SAR {hosted_cost} per 1,000 requests. The chamber's own machine: SAR {monthly} a
month, which is SAR {owned_cost} per 1,000 at the expected volume and SAR {capacity_cost} per 1,000 if
the machine were full ({measured_here}; the throughput is the course model's and must be measured again
on the real model and accelerator).

## Licence

{licence_line}

## Data placement

{placement_line}

## What would change this decision

A paired comparison on real requests that differs from the one above; a regression found after
release; a change in the classification of the requests; a second service that shares the machine.
