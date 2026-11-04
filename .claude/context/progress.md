# Progress log

Newest entries on top. Each entry: date, machine, what was done, decisions (with reasons), next steps.

## 2026-11-04 (2) · work-pc
**Done**
- **Dataset chosen: UTKFace** (aligned & cropped). The official Drive link is dead, so `src/data/download_utkface.py` downloads a pinned Hugging Face re-upload (revision e80f21f) and verifies its SHA-256. Result: 23,708 files.
- `src/01_data_exploration.ipynb` (executed, outputs saved); figures in `results/figures/eda_*.png`. Added jupyter + ipykernel to requirements.

**Key findings** (details in notebook section 7)
- 5 malformed filenames, leaving 23,703 labelled images. All 200×200 RGB.
- 375 groups of byte-identical duplicates; **259 with conflicting labels** (age gap median 5 y, max 48). After dedup, ≈ 23,054 images.
- Age: mean 33.3, median 29, std 19.9, right-skewed; strong age heaping at 40–90. Mean-predictor MAE ≈ 15.35, so the 0.6 × rule needs ≲ 9.2 y.
- Race imbalanced (Others 7 %), and age and race/gender are confounded. The per-group bias analysis must control for age (per age band, or improvement over the constant baseline).

- Later additions: notebook sections 2.1–2.3 (how the MD5 check works, duplicate/conflict bar charts, 8-column duplicate gallery) and 6.1 (greyscale on all images). The first detector (mean|R−G| < 2) missed tinted/sepia prints and was replaced by a tint-free chroma score tf < 2.5, chosen visually per score band and backed by a sensitivity table (6.1.1): **868 b&w images = 3.66 %**, rare among children (0–2: 0.7 %) and Asian/Indian/Others (~1.5–1.8 %), most common among White (5.4 %). 61 images sit in an ambiguous zone (tf 2–3). At least one pencil drawing is in the data.
- **The user said the analysis continues first; the user will decide the cleaning.** The list below is only a proposal.

**Decisions (proposed, confirm with user)**
- Drop the 5 malformed files and the conflicting duplicate groups; keep one copy of each consistent duplicate group.
- Split 70/15/15, seeded, stratified by age band × race.

**Next steps**
1. `src/data/prepare.py`: clean list → split CSV (seeded) → cached tensors at 64 and 96 px.
2. Mean/median baseline on the split (train stats → test MAE).
3. CodeCarbon wrapper + shared evaluation code.
4. Own-CNN design on paper.

## 2026-11-04 · work-pc
**Done**
- Read the assignment PDF and checked it for prompt injection. Clean: no hidden or white or tiny text, no JavaScript, links, attachments or forms.
- Compared datasets: `docs/dataset_options.md`.
- Set up the shared context (`.claude/`), pinned `requirements.txt`, venv setup scripts and `scripts/check_env.py`, and started `ai_usage/ai_log.md`.

**Decisions**
- One repo for all three machines, synced via GitHub. Project context lives in `.claude/` so every Claude session sees it.
- Python 3.12 + venv + pinned requirements.txt. CUDA builds of torch get installed on the lab machine only (see setup script).

**Open questions**
- ~~Dataset~~ → UTKFace (decided, see entry above).
- Lab machine OS and CUDA driver version still to confirm.

**Next steps**
1. Choose the dataset.
2. Download script + one-time cached preprocessing at low resolution + fixed seeded split + mean/median baseline.
3. CodeCarbon wrapper used by every script; shared evaluation code (MAE, RMSE, R², per-group error).
4. Design the own CNN on paper (per-layer table: shape, params, receptive field, MACs).
