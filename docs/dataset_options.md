# Dataset options (compared 2026-11-04)

The two "acceptable" rules from Section 3.5 decide most of this:

- **R1:** MAE_own ≤ 0.6 × MAE_mean. The target has to be clearly learnable.
- **R2:** MAE_own ≤ 1.15 × MAE_best_standard. A small CNN trained from scratch has to come close to an ImageNet-pretrained model. This is the real risk: the more ImageNet features help on a task, the harder R2 gets.

Footprint is graded as well (25 pts), so the best dataset is one that is cheap to iterate on at low resolution.

| Dataset | Images | Target | Licence / access | R1 (beat mean) | R2 risk (scratch vs. pretrained) | Bias analysis | Energy / effort |
|---|---|---|---|---|---|---|---|
| **UTKFace** (aligned & cropped) | 20k+ | age 0–116 | non-commercial research; ~107 MB zip (Google Drive, Kaggle mirrors) | easy (mean ≈ 15 y, CNNs ≈ 5–7 y) | **medium**: pretraining helps on faces, so the gap is maybe 10–25 % | required: gender + 5 race groups + age bands, all in the filename | low: already cropped, 64–128 px is enough |
| **BBBC005** synthetic cells | 19,200 TIF (696×520, grey) | cell count 1–100 | **CC0** (public domain); 1.8 GB zip | very easy | **low**: ImageNet features matter little on synthetic blobs, so a tiny CNN can match or beat them | not applicable (no people) | very low: grey, can shrink a lot. Counts likely take only ~24 distinct values; heavy blur (F up to 48) makes some images ambiguous |
| **RSNA Bone Age** | 12,611 train (+1,425 val) | age in months | RSNA terms: academic/non-commercial with attribution; 9.6 GB | easy (mean ≈ 33 mo) | **medium–high**: needs higher resolution to work well | sex (M/F) | **high**: large X-rays, so caching at low resolution is a must |
| **APPA-REAL** | 7,591 | real + apparent age | ChaLearn download; licence text to check | easy | **high**: only 4.1k train images, so pretraining wins clearly | gender, ethnicity (extended annotations) | low, but has known noisy crops |
| BIWI head pose | ~15k frames, 20 people | yaw/pitch/roll | research-only, mirrors | easy | medium | weak (20 subjects) | medium: multi-output, needs a subject-wise split and face crops |
| PetFinder Pawpularity | 9,912 | score 1–100 | Kaggle competition rules | **fails R1**: best models barely beat the mean | – | – | – |
| AVA / KonIQ aesthetics | 10k–255k | mean score | mixed | **likely fails R1** | high | – | – |
| ShanghaiTech, FG-NET, AFLW2000 | < 2,000 each | – | – | – | – | – | under the ~2,000-image minimum |
| AgeDB | 16,488 | age | must sign a licence by e-mail | – | – | – | approval delay is a risk |

## Recommendation

1. **UTKFace (first choice).** It's the most "standard" choice for the report. The related work is easy to cite, the bias analysis gives the discussion real substance, and it's cheap at 64–128 px. R2 is the risk. Mitigations: a well-designed small residual / depthwise-separable net, augmentation, and knowledge distillation from the fine-tuned model (allowed by Section 3.4). Distillation keeps the student "from scratch", since it starts from random initialisation.
2. **BBBC005 (safe, greenest).** It almost guarantees an acceptable model with a tiny footprint, the licence is the cleanest (CC0), and there's no sensitive data. The downsides: it's less interesting for the discussion, and the discrete count levels have to be handled carefully (stratified split, and report error per blur level as the "group" analysis).

Decision: _pending (user)._
