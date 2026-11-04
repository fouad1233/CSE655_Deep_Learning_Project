# Report notes (things the technical report must mention)

Collected while working, so nothing is forgotten when the report is written. Numbers come from `src/01_data_exploration.ipynb`.

## Dataset and cleaning (Section: problem and dataset)
- UTKFace aligned & cropped, 23,708 files (200×200 RGB). Source: the official Google Drive link is dead, so we used a Hugging Face re-upload pinned by revision and verified by SHA-256. Cite Zhang, Song & Qi (CVPR 2017); non-commercial research licence.
- Label noise found by MD5 hashing: 375 groups of byte-identical images, of which 259 have conflicting labels (age gap up to 48 years). Limitation: only exact copies are detected, not re-encoded near-duplicates.
- Cleaning (notebook section 8):
  - R1: drop 5 malformed filenames.
  - R2: drop all 531 images of conflicting duplicate groups.
  - R3: keep one copy of each consistent group (118 dropped).
  - R4: **"One non-photographic image (a pencil drawing) was removed after visual inspection."** The dataset may contain a few more such images; we did not inspect all 23k images by eye.
  - Result: 23,053 images. Label distributions change by at most ~0.2 percentage points.
- Greyscale: 868 images (3.66 %) are black-and-white (tint-free chroma score tf < 2.5; a naive |R−G| test missed sepia prints). **Kept**: valid faces and labels, and removing them would shift the race balance. Shortcut risk ("b&w means not a child", since only 0.7 % of ages 0–2 are b&w) handled by random-greyscale augmentation and measured by reporting the error on b&w vs. colour test images.

## Bias analysis (Section: bias)
- Age and group are confounded (0–2 band 39 % Asian; 80+ band 72 % White; women dominate ages 3–29, men 40–69). The constant predictor already gives very different per-group MAE (Black 10.6 vs. White 17.9). So report per-group MAE **within age bands** and/or the improvement over the constant baseline per group.
- Strong age heaping at 40–90 (multiples of 10 occur 2.4–5.9× more often than their neighbours): many labels of older people are estimates, which puts a floor on the reachable error.
