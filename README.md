# CSE655 Term Project: Green CNNs for Age Regression on UTKFace

Fouad Aladhami (254201001408), M.Sc. Program in Artificial Intelligence, Gebze Technical University, Fall 2026.

Predict a person's age from a 200×200 face image, compare our own CNN (trained from scratch) with two fine-tuned
ImageNet models, and measure the energy and CO2 of every run with CodeCarbon. Assignment: `assignment/`.

**Result:** our final CNN (2.80 M parameters, 374 M MACs) reaches a test MAE of **5.140** years, acceptable under the
assignment's rule (limit 1.15 × 4.768 = 5.483, best standard model = ResNet18 full fine-tuning). The whole project used
72.0 Wh and emitted 33.5 g CO2eq (72 tracked runs).

| Deliverable | Where |
|---|---|
| Technical report (PDF + LaTeX source) | `report/main.pdf`, `report/main.tex`, `report/refs.bib` |
| AI Usage Report + log + chat export | `ai_usage/ai_usage_report.pdf`, `ai_usage/ai_log.md`, `ai_usage/transcripts/` |
| Code | `src/` (notebooks, model, data scripts), `scripts/` |
| CodeCarbon logs | `logs/emissions/emissions.csv` (raw CodeCarbon), `logs/runs.csv` (one row per run) |
| Results table | `results/results.csv`, per-group errors in `results/per_group/` |
| Final weights of our CNN | `weights/final/own_cnn_d2_run02.pt` (11.2 MB) |

## 1. Setup (Python 3.12)

```bash
bash scripts/setup_env.sh            # macOS / Linux: creates .venv and installs requirements.txt
source .venv/bin/activate
python scripts/check_env.py          # versions, device, and a real GPU matmul
```
Windows: `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`.
On an RTX 50-series GPU (sm_120) torch needs a CUDA >= 12.8 build; the lab machine used
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 bash scripts/setup_env.sh` (exact versions: `env/freeze-lab.txt`).

All reported runs were made on the lab machine: AMD Ryzen 9 9950X3D, NVIDIA RTX 5090 (`cuda:0`), Ubuntu 24.04,
PyTorch 2.15 + CUDA 13.2, CodeCarbon 3.3.2 (country TUR, 465 g CO2eq/kWh). Seed 655 everywhere.

## 2. Data, cleaning and split

```bash
python src/data/download_utkface.py          # pinned Hugging Face copy of UTKFace, SHA-256 checked -> data/raw/
python scripts/run_notebook_cells.py src/01_data_exploration.ipynb --deps 1,3,13 --run 40,42,44,46   # cleaning -> data/processed/utkface_clean.csv (23,053 images)
python src/data/split.py                     # seeded 70/15/15 split -> splits/utkface_split_seed655.csv (committed)
```
`src/01_data_exploration.ipynb` is the full exploration (duplicates, labels, groups, greyscale photos) and the cleaning
rules R1–R4 (section 9). UTKFace: Zhang, Song & Qi, CVPR 2017; non-commercial research use only.

## 3. Training and evaluation: `src/02_pipeline.ipynb`

Run the notebook in order (`jupyter nbconvert --to notebook --execute --inplace src/02_pipeline.ipynb`), or single
sections with `python scripts/run_notebook_cells.py src/02_pipeline.ipynb --list` / `--deps … --run …`.
Every training, evaluation and test run is wrapped in `track()` (CodeCarbon) and logged to `logs/runs.csv`.

**Re-running from scratch:** every measured cell is guarded by `already_logged(run_name)`, so with the committed logs
the notebook only re-displays the results and starts no run (this protects the test set from being used twice).
To repeat the measurements, move `logs/runs.csv` and `logs/emissions/emissions.csv` away first. A full re-run makes
10 training runs and 6 short learning-rate runs and costs about 72 Wh (under 15 minutes of GPU time on one RTX 5090).
Re-runs give close but not bit-identical numbers: bf16 mixed precision and cuDNN's autotuned kernels are not
deterministic on the GPU, so expect MAEs within a few hundredths of a year and energies within a few percent.

| Notebook section (code cell) | What it does | Outputs |
|---|---|---|
| 1–4 (cells 2–8) | setup, split check, CodeCarbon helper `track()`, uint8 image cache on the GPU | `data/cache/` |
| 5–6 (10, 12) | regression heads, `evaluate()`, mean/median baseline | `results/results.csv` (baselines) |
| 7–9 (14–20) | augmentation, freezing plan (option A), training loop, smoke test | |
| 10–11 (22–28) | ResNet18 and MobileNetV3-L, strategy A and full fine-tuning | `logs/train/`, Fig. 10 |
| 12 (30, 32) | single test pass of the 4 standard runs, latency, energy per 1,000 predictions | `results/results.csv`, `results/per_group/`, Fig. 9 |
| 13 (34) | signed error per age band (validation set) | Fig. 8, `results/error_by_age_val.csv` |
| 14 (36) | balanced-loss ResNet18 run (validation only) | Fig. 13 |
| 15.1–15.3 (38–46) | own CNN shape check, smoke test, learning-rate check on 4,000 images | Fig. 11 |
| 15.4–15.6 (48–52) | own CNN v1 (40 epochs), bottleneck ablation, their single test passes | Fig. 5, Table III |
| 15.7–15.8 (54, 56) | final CNN (2 blocks/stage, 60 epochs), hook-checked per-layer table, single test pass | Fig. 3, Fig. 12, Table II |
| 15.9 (58) | test MAE by age band | Fig. 7 |

## 4. Report tables and figures

After the notebook, run the two plotting/aggregation scripts (CSV arithmetic and plotting only, no model is run):
```bash
python scripts/update_results_footprint.py   # train_s, project_kwh, project_kg_co2 in results/results.csv (sums of logs/runs.csv)
python scripts/plot_error_vs_co2.py          # Fig. 6
python scripts/draw_report_archs.py          # Fig. 2 and Fig. 4 (architecture diagrams from forward hooks)
python scripts/model_facts.py results/model_facts.json   # params, MACs at 224/200/128 px, CPU latency of the candidate models (CPU only, not tracked)
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

| In the report | Produced by |
|---|---|
| Table I (cleaning, split) | `01_data_exploration.ipynb` section 9, `src/data/split.py` |
| Table II (every layer of the final CNN) | `02_pipeline.ipynb` cell 54 (`layer_table` + hook asserts) |
| Table III (ablation) | `results/results.csv`, `logs/runs.csv` (cells 48–52) |
| Table IV (main results) | `results/results.csv` after `update_results_footprint.py` |
| Table V (per gender / race) | `results/per_group/*_test.csv` (cells 12, 30, 52, 56) |
| Fig. 1 age histogram | `01_data_exploration.ipynb` section 4 → `results/figures/eda_age_hist.png` |
| Fig. 2, Fig. 4 architectures | `scripts/draw_report_archs.py` → `results/figures/arch_report/` |
| Fig. 3 own CNN v1 vs final | cell 54 → `own_cnn_d2_vs_run01.png` |
| Fig. 5 ablation | cell 50 → `own_cnn_ablation_bottleneck.png` |
| Fig. 6 error vs CO2 | `scripts/plot_error_vs_co2.py` → `error_vs_co2_final_and_project.png` |
| Fig. 7 MAE by age (test) | cell 58 → `test_mae_by_age_own_vs_standard.png` |
| Fig. 8 age bias (validation) | cell 34 → `error_by_age_val.png` |
| Fig. 9 standard-model findings | cell 32 → `findings_standard_models.png` |
| Fig. 10 standard training curves | cell 28 → `compare_standard_runs.png` |
| Fig. 11 learning-rate check | cell 46 → `own_cnn_lr_search2.png` |
| Fig. 12 final CNN training run | cell 54 (`train_model`) → `train_own_cnn_d2_run02.png` |
| Fig. 13 balanced loss | cell 36 → `balanced_loss_vs_plain_val.png` |
| Project totals (72.0 Wh, 33.5 g) | `python scripts/update_results_footprint.py` (last line) |
| ResNet18 MACs at 128 px (0.59 G, §III-C) | `scripts/model_facts.py` → `results/model_facts.json` |

## 5. Using the final model

```python
import torch
from src.models.own_cnn import OwnCNN
model = OwnCNN(depths=(2, 2, 2, 2))
model.load_state_dict(torch.load("weights/final/own_cnn_d2_run02.pt", map_location="cpu"))
model.eval()
# input: float tensor (B, 3, 200, 200), pixels / 255, normalised with the ImageNet mean and std; output: age in years
```
