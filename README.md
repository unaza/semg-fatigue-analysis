# Multichannel sEMG Muscle Fatigue Analysis

Epoch-based RMS/MDF analysis of 8-channel forearm surface EMG (15 subjects) during a sustained isometric hold, followed by subject-independent classification of early versus late task states. Undergraduate Biomedical Engineering semester project. Author: Unazza Asif

> **Provenance:** the dataset, acquisition protocol and the core RMS/MDF slope method come from Ebied et al. (2020). This repository is my Python re-implementation of that foundation plus additional analyses built around it.

## Relationship to the Original Study

| | Content |
|---|---|
| **From Ebied et al.** | Dataset and protocol; RMS and MDF per 1 s epoch; ΔRMS / ΔMDF as slopes of a first-order fit; one-sample t-tests across channels and subjects |
| **My reproduction** | The same analysis in Python. Qualitatively consistent with the paper (ΔMDF mostly negative, ΔRMS mostly positive, Ch4 most MDF-affected, Ch3 the only channel without a significant RMS change). Differences: Ch1 is also non-significant for MDF, Sub3 is significant, and the paper's envelope step is not implemented |
| **My extensions** | Signal-quality checks; R² of the linear fits; Wilcoxon tests, Holm/BH correction and CIs; alternative spectral indices (notch-excluded MDF, log SMR, log L/H); fatigue-zone analysis; piecewise/exponential trajectory models with onset estimates; leave-one-subject-out (LOSO) classification with baseline normalization, permutation and sign-flip tests, a three-class variant, and channel analyses |

Standard methods (RMS, MDF, t-tests, logistic regression, random forest, LOSO) are not claimed as novel. Details: [docs/METHODS.md](docs/METHODS.md).

## Pipeline

```
Raw 8-channel sEMG → quality control → 1 s epochs (RMS, MDF, other features)
→ slopes and statistics → spectral-index, fatigue-zone and trajectory/onset analyses
→ 5 s windows + per-subject baseline normalization (first 10 s)
→ LOSO classification (LR, RF, gradient boosting) → permutation / sign-flip validation
```

## Dataset and Attribution

I did not collect the data. The dataset is from the Zenodo record below (CC BY 4.0), and the study that used it is cited after it.

> A. Ebied, A. M. Awadallah, M. A. Abbass, and Y. El-Sharkawy, "Multi-channel Surface EMG Dataset for Fatigue analysis," Zenodo, v1, 2021, doi:10.5281/zenodo.5189275. <https://zenodo.org/records/5189275>

> A. Ebied, A. M. Awadallah, M. A. Abbass, and Y. El-Sharkawy, "Upper Limb Muscle Fatigue Analysis Using Multi-channel Surface EMG," in *Proceedings of NILES2020: 2nd Novel Intelligent and Leading Emerging Sciences Conference*, IEEE, 2020, pp. 423–427, doi:10.1109/NILES50944.2020.9257909.

Per the paper: 15 healthy young males, Myo armband (8 channels, 200 Hz), 6 kg isometric hold for 120 s. The raw data are not included in this repository; download them from Zenodo (see [data/README.md](data/README.md)).

## Key Results

Labels come from time within the recording, so "early vs late" is a **proxy** for fatigue state, not clinical fatigue detection.

| Signal analysis | Result |
|---|---|
| Mean MDF, first 10 s → last 10 s | 63.4 → 56.3 Hz |
| Overall ΔMDF / ΔRMS | −0.0590 Hz/s [−0.0765, −0.0416] / +0.1010 RMS units/s [0.0709, 0.1312] |
| Series with expected slope sign | ΔRMS > 0: 90.0%; ΔMDF < 0: 85.0% |
| Median linear-fit R² | RMS 0.453; **MDF 0.087** (54.2% of MDF series have R² < 0.1) |
| Channels significant after Holm | ΔMDF 6/8 (not Ch1, Ch2); ΔRMS 7/8 (not Ch3) |

| Classification (LOSO, 15 held-out subjects) | Result |
|---|---|
| Binary task | 630 windows; majority-class accuracy 0.619 |
| Pre-specified: Combined features, logistic regression, normalized | Balanced accuracy 0.929; permutation p = 0.001; sign-flip p = 0.00003 |
| Highest scoring (chosen after seeing results): Spectral-index, random forest | Balanced accuracy 0.954, AUC 0.993 |
| Without baseline normalization | Best balanced accuracy 0.739; normalization raised accuracy by about 0.22–0.24 (Holm p ≤ 0.024) |
| Three-class (early/middle/late) | Best balanced accuracy 0.693 (chance ≈ 0.33) |

## Selected Figures

![Mean slopes per subject and channel with 95% CIs](results/figures/mean_slopes_ci.png)
![Linear-fit R² for RMS and MDF](results/figures/linear_fit_r2_histograms.png)
![Binary LOSO accuracy by feature set](results/figures/binary_accuracy_by_feature_set.png)
![Channel contribution analysis](results/figures/channel_contribution_analysis.png)

Four more figures are in `results/figures/`; key tables are in `results/tables/`.

## Limitations

- Only 15 healthy young males, one task and one consumer-grade device (8-bit, 200 Hz, 0.2–3.5% of samples at the amplitude limits); no external validation.
- No independent fatigue measure (force, MVC, time to failure); MDF trends are weak within individual series.
- No band-pass filtering beyond the device's built-in 50 Hz notch (following the paper); sensitivity to filtering and PSD method was not tested.
- The best results need a 10 s per-subject baseline; windows overlap, so they are not independent.
- The highest-scoring configuration was picked from the same LOSO results, so its score is optimistic.
- Fatigue-zone scores, onset times and the subject ranking are exploratory and unvalidated.

## Repository Structure

```
├── EMG_Fatigue_Analysis_Clean.ipynb   # complete analysis
├── requirements.txt
├── README.md
├── LICENSE                            # MIT (code only)
├── .gitignore
├── CITATION.cff
├── data/README.md                     # where to place the dataset (data itself not included)
├── docs/METHODS.md                    # detailed methods and results
└── results/
    ├── figures/                       # 8 key figures
    ├── tables/                        # key tables and summary
    └── supplementary/                 # full intermediate tables
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook EMG_Fatigue_Analysis.ipynb   # run all cells
```

Place `All_15sub_Data.mat` in `data/` (or set `MAT_PATH`). The permutation test is the slowest step.

## Citation and License

If you reuse this work, please cite Ebied et al. and the Zenodo dataset (above), and acknowledge that the dataset and the RMS/MDF slope methodology originate from that study.

**License:** the code is released under the MIT License (see [LICENSE](LICENSE)). The dataset is licensed separately by its authors under CC BY 4.0 and is not covered by the code license.
