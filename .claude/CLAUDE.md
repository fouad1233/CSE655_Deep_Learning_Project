# CSE655 Green-CNN term project: shared context

This file is committed and loads automatically in every Claude Code session on every machine.
Everything durable about the project lives **in this repo**, not in machine-local Claude memory. That way the work PC, the Mac, and the lab machine all see the same state after `git pull`.

## Start of every session
1. `git pull`.
2. Read `.claude/context/progress.md` (status, decisions, next steps).
3. Use the project venv (`.venv`). Never install packages globally.

## End of every session (or after any important step)
- Update `.claude/context/progress.md`: what was done, decisions with reasons, next steps.
- Append the important AI interactions to `ai_usage/ai_log.md`, **including mistakes the AI made** (the report needs at least 3).
- Remind the user to commit and push. Don't commit or push unless asked.

## The assignment (full text: `assignment/CSE655_Term_Project_Green_CNN_Regression.pdf`)
- Individual project. Image regression with our own CNN trained from scratch, compared with 2 standard models (one large, one efficiency-oriented; at least one fine-tuned from ImageNet).
- Goal: an **acceptable** model with the **lowest total CO2**. Acceptable means MAE_own ≤ 1.15 × MAE_best_standard **and** MAE_own ≤ 0.6 × MAE_mean.
- **Every** training and evaluation run is tracked with CodeCarbon, including failed runs and hyperparameter searches. Never run training without the tracker.
- Fixed seeded train/val/test split shared by all models. The test set is used once per final model and never for tuning.
- Deliverables: report (≤ 10 pages), AI Usage Report (2–5 pages), code + README + requirements.txt, all emissions.csv files + results.csv, final weights.
- Deadline **Fri 2026-11-27 23:59**. Sit-in oral exam **Tue 2026-12-01**: hand-compute shapes, parameter counts and receptive fields, and explain any line of code or number.

## How to work with the user
- The user must be able to defend every line and number in the exam. Explain design choices and show hand calculations next to code; don't just hand over code.
- Every run costs CO2 and counts against the footprint grade. Think before running, test on tiny subsets, and avoid "just to be sure" runs.
- Keep code identical across machines; only `--device` / config values may differ.
- **The user drives the phases.** Don't move to the next step (e.g. cleaning, split, training) on your own. Finish the current analysis and let the user decide what comes next and which cleaning rules to apply.
- **Notebooks: run only the new/changed cells**, not the whole notebook: `python scripts/run_notebook_cells.py NB --list`, then `--deps <cells that build the needed state> --run <new cells>`. Re-run everything only if the user asks.
- Git identity for this repo is set **per repo** (`git config user.name fouad1233`). The global config on the work PC belongs to the user's work repos; don't touch it.

## Machines (one repo, synced via GitHub: fouad1233/CSE655_Deep_Learning_Project)
| Name | Hardware | Use | torch device |
|---|---|---|---|
| work-pc | Windows 11, i5-1135G7, 16 GB, no NVIDIA GPU | code, data prep, small CPU runs | `cpu` |
| mac | MacBook M1 | code, small runs | `mps` |
| lab | Linux (assumed), 2× RTX 5090 (Blackwell, needs CUDA ≥ 12.8 build) | heavy training | `cuda` |

Record the machine name in every run's log. CodeCarbon numbers depend on the hardware, and the report must say where each run happened.

## Conventions
- Python 3.12 (`.python-version`), venv at `.venv`, pinned `requirements.txt`. Setup: `scripts/setup_env.sh` (mac/linux) or `scripts/setup_env.ps1` (windows), then check with `python scripts/check_env.py`.
- Data lives in `data/` (gitignored). Each machine downloads it with the same script, so the split is reproducible from the seed.
- Logs: `logs/emissions/` (CodeCarbon CSVs, committed); results: `results/results.csv`.
