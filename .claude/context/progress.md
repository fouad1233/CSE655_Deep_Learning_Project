# Progress log

Newest entries on top. Each entry: date, machine, what was done, decisions (with reasons), next steps.

## 2026-11-04 · work-pc
**Done**
- Read the assignment PDF and checked it for prompt injection. Clean: no hidden or white or tiny text, no JavaScript, links, attachments or forms.
- Compared datasets: `docs/dataset_options.md`.
- Set up the shared context (`.claude/`), pinned `requirements.txt`, venv setup scripts and `scripts/check_env.py`, and started `ai_usage/ai_log.md`.

**Decisions**
- One repo for all three machines, synced via GitHub. Project context lives in `.claude/` so every Claude session sees it.
- Python 3.12 + venv + pinned requirements.txt. CUDA builds of torch get installed on the lab machine only (see setup script).

**Open questions**
- Dataset: UTKFace (recommended) vs. BBBC005 (safest / greenest). Waiting for the user.
- Lab machine OS and CUDA driver version still to confirm.

**Next steps**
1. Choose the dataset.
2. Download script + one-time cached preprocessing at low resolution + fixed seeded split + mean/median baseline.
3. CodeCarbon wrapper used by every script; shared evaluation code (MAE, RMSE, R², per-group error).
4. Design the own CNN on paper (per-layer table: shape, params, receptive field, MACs).
