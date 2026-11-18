# Progress log

Newest entries on top. Each entry: date, machine, what was done, decisions (with reasons), next steps.

## 2026-11-18 · lab
**Done (setup only; no training, no CO2 runs):**
- Fresh clone; repo-local git identity set (`fouad1233`).
- Lab: Ubuntu, kernel 7.0, 2× RTX 5090 (sm_120), driver 580.178.04 (CUDA 13.0), Python 3.12.13 (`~/.local/bin/python3.12`).
- **torch index:** `cu128` stops at torch 2.11, so it has no build for the pinned `torch==2.15.0`. Only `cu132` has 2.15.0 for cp312. Installed with `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 bash scripts/setup_env.sh` → `torch 2.15.0+cu132`, `torchvision 0.29.2+cu132`. The 13.2 runtime works on the 13.0 driver (CUDA minor-version compatibility). Verified with `check_env.py` (arch list includes sm_120, both GPUs seen, conv smoke test OK) and a real 4096² matmul on each GPU. Full resolution in `env/freeze-lab.txt`.
- UTKFace downloaded (SHA-256 OK, 23,708 files); clean CSVs regenerated → `utkface_clean.csv` with **23,053 rows** (matches work-pc). Re-running the cells only changed notebook metadata (timestamps, image path) and re-rendered `clean_step1_removed.png`, not the results.

**Decided (user, 2026-11-18):**
- **Split 70/15/15, stratified by age band × race**, seed 655.
- **Input size 200 px (native UTKFace)** for all models; no resizing. ResNet18 ≈ 1.56 G MACs/img at 200 px.
- **Standard models: ResNet18 + MobileNetV3-Large, both fine-tuned from ImageNet.**

**Done:** `src/data/split.py` → `splits/utkface_split_seed655.csv` (committed; MD5 573aa142…, identical on re-run). Hand-rolled per-stratum split (shuffle, round(0.15 n) test, round(0.15 n) val, rest train), because 2 strata are tiny (70–79 Others n=2, 80+ Others n=4) and sklearn's stratify fails on them. Result: train 16,131 / val 3,461 / test 3,461; age mean 33.23 / 33.26 / 33.33, race shares within 0.1 pp.

**Notebook `src/02_pipeline.ipynb`** (built cell by cell with the user; each step explained and confirmed before the next):
1. Setup: `REPO`, `SEED=655`, `MACHINE="lab"`, real matmul GPU check.
2. Load + check the split.
3. `track(run_name, kind, gpu_ids)`: CodeCarbon `OfflineEmissionsTracker(country_iso_code="TUR")`, **`tracking_mode="process"`, `force_mode_cpu_load=True`**. Reason: the lab machine is **shared** (another user's Ollama on **GPU 1** + PX4 sims ≈ 120 W CPU); RAPL would charge their CPU load to our runs. GPU measured by NVML on `gpu_ids=[0]` only; `track` warns/logs `other_gpu_procs` if anything else is on our GPU. Logs: `logs/emissions/emissions.csv` (CodeCarbon) + `logs/runs.csv` (our summary; read+concat+rewrite, never append). Failed runs logged with `status=failed`. Turkey grid: 465 g CO2eq/kWh. **Always train on cuda:0.**
4. Cache: `data/cache/utkface_200_uint8.npy` (23,053 × 3 × 200 × 200 uint8, 2.77 GB, rows = split CSV order), built in 6.2 s / 0.146 Wh; 8 random rows equal a fresh JPEG decode.
5. `build_model("resnet18" | "mobilenet_v3_large")`: ImageNet **V1 weights for both** (user decided MNV3 V1), last Linear → 1 output. Params 11,177,025 / 4,203,313 (hand-checked). `to_input`: uint8 → /255 → ImageNet mean/std on GPU. ResNet18 at 200 px: 1.56 G MACs.
- User ran `sudo chmod o+r` on the RAPL `energy_uj` files; no longer needed (process mode). Resets on reboot.

6. `evaluate(y, ŷ, meta)` → MAE, RMSE, R² + per-group MAE (gender, race, age band); `save_result(row)` → `results/results.csv` (one row per model, replaced on re-run); per-group CSVs in `results/per_group/`.
   **Baseline (train stats, evaluated on val + test):** mean = 33.228 → **test MAE 15.400**, RMSE 20.007, R² −0.000; median = 29 → test MAE 15.072, RMSE 20.471. **MAE_mean = 15.400 ⇒ own CNN target ≤ 9.240 years** (and ≤ 1.15 × MAE_best). Mean-baseline per-group test MAE ranges from 2.51 (30–39) to 54.36 (80+); by race Black 10.60 (youngest group) … White 17.89. These differences come from the age distribution, not the images, so bias analysis must compare models **per age band** or against this baseline.

7. Train/val uint8 on cuda:0 (2.35 GB); `augment()`: flip p=0.5, rotation ±10° + shift ±8 px in ONE bilinear `grid_sample`, **black (zero) fill** (user: reflection would create fake ears/hair), greyscale p=0.1. Same augmentation for all 3 models.
8. Option A freezing (`TRAIN_PARTS`, `set_phase`, `set_train_mode`; frozen BN stays eval): phase 2 trains ResNet18 `layer3, layer4, fc` (10,493,953 params, est. 66 % of full-FT step cost) / MNV3 `features[7:]` + `classifier` (4,142,193, 80 %). Rule: train everything at 13×13 and 7×7. MACs at 200 px: ResNet18 1.556 G, MNV3-L 0.187 G (own hook counter; torchinfo block totals for MNV3 are wrong).
9. `train_model()` (rewritten for readability + line comments at user request): head bias = 33.23; phase 1 = 1 epoch head, AdamW 1e-3; phase 2 = AdamW head 1e-3 / backbone 1e-4, wd 0.01, cosine to 0 over 14 epochs, bs 128, bf16 autocast, early stopping patience 3 on val MAE, best weights kept. Logs `logs/train/<run>.csv` + `_steps.csv`, figure `results/figures/train_<run>.png` (loss per step, log scale, time per epoch). **Smoke test passed** (512 train / 256 val, 2 epochs, both models, ~0.065 Wh each). Demo cells (heads, augmentation, freezing plan) now run once (`already_logged`).
- **User rule: never start a training run without the user's explicit OK.**

**Next (user decides when):** full training runs of ResNet18 and MobileNetV3-L (need the user's OK), then test evaluation once per final model. Earlier note: the training loop and the fine-tuning plan (open: head-first then unfreeze vs. full fine-tuning; loss L1/Huber; LR, epochs, early stopping, AMP, batch size, augmentation incl. random greyscale).

## 2026-11-05 · work-pc
**Done (deck, at the user's request; no training, no CO2 runs):**
- After a slide comment, slide 3 now defines every symbol (H, W, k, s, p, C_in, C_out, ⌊ ⌋, params, MAC) with a step-by-step ResNet18 conv1 example, and 4 glossary slides explain 40 terms in plain words.
- 6 new slides (15–20) count **every ResNet18 parameter by hand**: rules (conv k·k·C_in·C_out with bias=False; BN 2·C = γ, β, running stats are buffers; Linear C_in·C_out + C_out; ReLU/pool/⊕ = 0; stride and padding never change params), then stem+layer1, layer2, layer3, layer4 (PyTorch name, weight shape, formula, BN, sum) and the total. Deck is now 42 slides.
- Exam numbers (checked against `named_parameters()`, torchvision 0.29.2): stem 9,536 · layer1 147,968 · layer2 525,568 · layer3 2,099,712 · layer4 8,393,728 · fc 513 → **11,177,025** (conv 11,166,912 + BN 9,600 + fc 513). With the ImageNet head: 11,689,512. Shortcut: layer2–4 conv weights = 32·C² (C = output channels).

- User caught that the deck used 128 px everywhere without saying the standard (ImageNet) input is 224×224 (logged as AI mistake case 3). Facts for the input-size decision, measured on torchvision 0.29.2: ResNet18 params are identical at any size (AdaptiveAvgPool before fc); MACs with the 1-output head: 224 px 1.814 G, 200 px (UTKFace native) 1.556 G, 128 px 0.592 G. layer4 map: 7×7 at 224 and 200, 4×4 at 128. Pretrained weights' transforms: resize 256, crop 224.
- New deck section "Input size" (slides 10–13, deck now 46 slides): where 224 comes from (He et al. §3.4, quotes checked in the PDF), which layer types work at any H×W (only fc needs a fixed 512), how AdaptiveAvgPool2d((1,1)) bridges any size to 512 (torchvision resnet.py lines 205, 276–280), a shape trace at 224/200/128 with the H_out rule, and what changes (maps, MACs, memory, detail, pre-training scale; Touvron et al. 2019, arXiv:1906.06423) vs. what does not (params, layers, channels, weights, file).

**Handoff to the lab machine (2× RTX 5090), 2026-11-18:** the user continues there with the model-design phase. Setup on lab:
1. Fresh `git clone` (history was rewritten on 2026-11-04; an old clone must not be pulled/merged), then `git config user.name fouad1233` and `git config user.email fouadiadhami@gmail.com` (repo-local).
2. `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_env.sh` (Blackwell needs a CUDA ≥ 12.8 build; if that index has no wheel for the pinned torch, pick the matching cu12x/cu13x index), then `python scripts/check_env.py` (must list sm_120 and see 2 GPUs).
3. `python src/data/download_utkface.py` (SHA-256 checked), then regenerate the clean CSVs with the command in entry (3) below → 23,053 rows.
4. Record `lab` as the machine in every run log; CodeCarbon on every training/eval run, including tests.

**Still open:** the same user decisions as in entry (5) below (which two models, **input size: 224 / 200 / 128**, fine-tuning plan).

## 2026-11-04 (5) · work-pc
**Plan change (user):** train the **standard models first**, then the own CNN. Reasons: know MAE_best (the target) early, debug the pipeline on known-good models, learn a training recipe. Conditions agreed in discussion: build the shared pipeline first (split, cached images, eval, CodeCarbon); compare on **validation** MAE during development, test once at the end; same split, input size and training seed for all models; final runs on one machine (lab) so energy is comparable.
**Done:** slide deck "Standard CNN Architectures for Age Regression" (claude.ai artifact https://claude.ai/artifact/L4EgPavVkVXqQiDddgDfdx, 23 slides): every candidate architecture layer by layer, real learned kernels, torchvision code, comparison, recommendation. Numbers come from `scripts/model_facts.py` → `results/model_facts.json` (torchvision 0.29.2).
- Later: paper-style 3D architecture diagrams (tensors as cuboids, kernels drawn at true scale on their input, shapes and params under every layer, residual/concat/SE arcs). Renderer `src/viz/arch_diagram.py` (reusable for the **own CNN** figure in the report); `scripts/draw_standard_archs.py` builds them from forward hooks → `results/figures/arch/*.png` (9 figures). Added as 9 slides; deck is now 32 slides.
**Recommendation (not yet decided by user):** fine-tune **ResNet18 (IMAGENET1K_V1)** + **MobileNetV3-Large**; runner-up EfficientNet-B0.
**Open decisions for the user:** (1) input size for all models (128 px suggested), (2) fine-tune both vs MobileNetV3 from scratch, (3) MobileNetV3-L V1 (74.0 %, 600-epoch recipe) vs V2 (75.3 %), (4) head-first then unfreeze vs full fine-tuning.

## 2026-11-04 (4) · work-pc
**Done:** notebook restructured at the user's request:
- "Look at the images" moved to the top as **section 2**: 2.1 image dimensions / tensor shapes (all 23,703 images are 200×200 RGB; NumPy (H,W,C) = (200,200,3), torch (C,H,W) = (3,200,200), 120,000 values) with a memory/compute table per input resolution; 2.2 random examples; 2.3 the same faces at 200/128/96/64/48/32 px (`results/figures/eda_resolutions.png`).
- Age bands are now defined in cell 3 (section 1). Everything after section 2 is renumbered (+1): duplicates 3, age 4, groups 5, baseline 6, greyscale 7, observations 8, **cleaning 9**.
- **Regenerate the clean CSVs:** `python scripts/run_notebook_cells.py src/01_data_exploration.ipynb --deps 1,3,13 --run 40,42,44,46`
- For the CNN design: input-resolution choice (64 vs. 96 vs. 128 px) is a key "green" ablation. Conv cost scales with H×W (64 px ≈ 0.10× of 200 px).

## 2026-11-04 (3) · work-pc
**Done:** cleaning step 1 (notebook section 8, now **section 9** after the restructure, decided by the user):
- R1: drop 5 malformed filenames. R2: drop all 531 images of the 259 duplicate groups with conflicting labels. R3: keep one copy (first filename alphabetically) of each of the 116 consistent groups, dropping 118 copies.
- Result: **23,054 clean images** (97.24 %) → `data/processed/utkface_clean.csv` (+ `utkface_removed.csv` with reasons). These are gitignored and regenerated by running notebook section 9 (command in entry (4) above).
- Distribution barely changed: the largest change in any gender/race/age-band share is 0.22 percentage points. R2 removed relatively more Indian (22.8 % vs. 16.8 %) and Others (12.8 % vs. 7.1 %) images.
- Added `scripts/run_notebook_cells.py` (run only selected cells).

**Decided later the same day (user):**
- **R4:** remove the pencil drawing `76_0_1_20170116205556715.jpg.chip.jpg` (not a photo), leaving **23,053 clean images**. **Mention it in the report** (one sentence; the user agreed after first wanting to leave it out).
- **Keep the greyscale images (868).** Handle the shortcut with random-greyscale augmentation and report the error on b&w vs. colour test images. Optional ablation: greyscale (1-channel) input for the own CNN (green angle).
- `docs/report_notes.md` started: facts and sentences the report must include.

**Still open:** anything else the user wants to analyse; no split yet.

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
