# Kaggle Workflow for Project 3

Use Kaggle as a remote GPU runner, not as a live mounted extension of the local workspace.

## Recommended Model

Local machine:
- source of truth for code
- git history
- Codex-driven edits
- result comparison

Kaggle:
- short GPU training/evaluation runs
- output artifact generation

## Practical Flow

1. Keep editing code locally in this folder.
2. Push the `IndiWASTE` dataset to Kaggle as a private Kaggle Dataset.
3. Fill in `kaggle_sync_config.json` from `kaggle_sync_config.example.json`.
4. Run `python kaggle_sync.py stage-kernel`.
5. Run `python kaggle_sync.py push`.
6. Run `python kaggle_sync.py watch`.
7. Pull `run.log`, `results.tsv`, and `last_eval.json` from `kaggle_outputs/`.
8. Let Codex compare the metric and decide whether to keep or revert.

## P100 Note

Some Kaggle GPU sessions still come up on `Tesla P100` hardware. The current Kaggle default
PyTorch build may not support that GPU architecture, so this sync layer enables internet access
and bootstraps a compatible `cu126` PyTorch build before `train.py` starts.

## Why This Is the Best Fit

Kaggle free GPU sessions are ephemeral. That makes them good for:
- running one experiment
- saving outputs
- shutting down

They are not ideal for:
- a permanent remote daemon
- direct interactive file editing from the local workspace
- long autonomous loops without explicit reruns

## Suggested Automation Level

Level 1:
- local editing
- manual Kaggle run
- local result review

Level 2:
- local editing
- Kaggle CLI push
- Kaggle run
- Kaggle output pull
- local result review

Level 3:
- fully scripted outer loop controlling Kaggle jobs from the local machine

For stability, start with Level 1 or Level 2.

## One-Time Local Setup

1. Install the Kaggle CLI:
   `python -m pip install kaggle`
2. Create a Kaggle API token from your Kaggle account settings.
3. Put `kaggle.json` in `%USERPROFILE%\\.kaggle\\kaggle.json`
4. Make sure `kaggle --version` works in your shell.

## One-Time Kaggle Dataset Setup

Upload the `IndiWASTE` dataset once as a private Kaggle Dataset.

Your config file should then point to:
- the Kaggle dataset source name
- the mounted dataset slug inside `/kaggle/input/...`
- the optional dataset subdirectory if your upload contains an `IndiWASTE/` folder

## Day-to-Day Commands

From this folder:

1. `python kaggle_sync.py prepare-config`
2. Edit `kaggle_sync_config.json`
3. `python kaggle_sync.py stage-kernel`
4. `python kaggle_sync.py push`
5. `python kaggle_sync.py watch`

After the run:

1. Inspect `kaggle_outputs/run.log`
2. Inspect `kaggle_outputs/last_eval.json`
3. Decide whether to keep or revert the local code change
