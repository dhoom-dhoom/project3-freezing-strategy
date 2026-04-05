# Project 3: Transfer Learning Layer Freezing Strategy

This is a Karpathy-style autoresearch setup for the DLCV Phase 2 Unit III project:

`Problem 14: Transfer Learning Layer Freezing Strategy`

The goal is to study:

`Which layers should you freeze when fine-tuning an ImageNet-pretrained ResNet on IndiWASTE?`

This setup follows the PDF guidelines:
- pretrained model instead of heavy from-scratch training
- one sharp research question
- minimal training with a short fixed budget
- controlled comparisons
- outputs that support the deliverable:
  - freezing strategy comparison table
  - recommendations

## Scope

Only `train.py` is the intended experiment surface.

`prepare.py` is fixed infrastructure:
- IndiWASTE dataset loading
- split handling
- transforms
- evaluation
- metric definition
- time budget constant

Do not modify `prepare.py` during the experiment loop unless the human explicitly asks.

## Research Objective

Compare freezing strategies for an ImageNet-pretrained ResNet on IndiWASTE under the same 300-second budget.

The default metric is:

- `val_error = 1 - val_macro_f1`

Lower is better.

We also log:

- `val_macro_f1`
- `val_accuracy`
- parameter counts

This gives us both a Karpathy-style scalar objective and the report-friendly classification metrics.

## Allowed Strategies

The intended strategies are:

- `all_but_head`
- `freeze_early`
- `freeze_late`
- `freeze_none`

The baseline should be `all_but_head`.

## Setup

1. Create a fresh branch named like `autoresearch/apr5-project3`.
2. Confirm the dataset exists at `../IndiWASTE` or set `INDIWASTE_ROOT`.
3. Confirm the project folder contains:
   - `prepare.py`
   - `train.py`
   - `program.md`
   - `results.tsv`
4. Initialize `results.tsv` if needed.
5. Run the baseline first with no code changes.

## Output Format

Every run ends with a summary like:

```text
---
val_error:         0.123456
val_macro_f1:      0.876544
val_accuracy:      0.882000
training_seconds:  300.0
total_seconds:     319.1
peak_memory_mb:    1234.5
num_steps:         145
num_params_M:      11.7
trainable_params_M: 0.0
freeze_strategy:   all_but_head
```

## Logging Results

Append every experiment to `results.tsv`:

```text
commit	val_error	val_macro_f1	val_accuracy	status	description
```

Status must be one of:

- `keep`
- `discard`
- `crash`

## Experiment Loop

LOOP FOREVER:

1. Read the current git state.
2. Form a freezing-strategy hypothesis.
3. Modify `train.py`.
4. Commit the change.
5. From this folder, run `uv run train.py > run.log 2>&1`
6. Extract `val_error`, `val_macro_f1`, and `val_accuracy` from the log.
7. If the run crashed, inspect the tail of the log and decide whether to fix or discard.
8. Record the result in `results.tsv`.
9. If `val_error` improved, keep the commit.
10. If `val_error` is worse or equal, reset back.

## Simplicity Rule

Prefer simpler changes when scores are effectively tied.

The purpose of this project is not to search over the whole training stack. It is to answer the Unit III question rigorously:

`Which freezing strategy should be recommended for IndiWASTE under a short fine-tuning budget?`
