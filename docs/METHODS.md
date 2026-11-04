# Detailed Methods and Results

Full technical write-up for the project. The short overview is in the repository [README](../README.md).

Epoch-based RMS/MDF fatigue analysis of 8-channel forearm surface EMG (15 subjects), followed by subject-independent classification of early versus late task states.

> **Provenance in one sentence:** the dataset, acquisition protocol and the core RMS/MDF slope methodology come from Ebied et al. (2020); this repository contains my Python re-implementation of that foundation plus additional analyses and a leave-one-subject-out machine-learning pipeline built around it.

Author: Unazza Asif, Riphah International University, Course: Artifical Intelligence

---

## Overview

Muscle fatigue during a sustained contraction is commonly associated with an increase in surface-EMG amplitude and a shift of the power spectrum toward lower frequencies. Ebied et al. quantified these changes in forearm muscles with two slope parameters, ΔRMS and ΔMDF, and used them to compare channels and subjects.

This project was developed in two stages:

1. **Reconstruction** of the Ebied et al. analysis (per-epoch RMS and median frequency, first-order slopes, one-sample t-tests across channels and subjects) on the dataset associated with their study.
2. **Extension** of that foundation into a broader pipeline: signal-quality checks, goodness-of-fit and multiple-comparison-corrected statistics, alternative spectral indices, fatigue-zone analysis, trajectory/onset modelling, and subject-independent classification of early versus late temporal windows with permutation-based validation.

The recordings contain **no independent measure of fatigue** (no force, MVC, time to task failure or perceived exertion). "Early" and "late" are defined by position in the recording and are used as a **proxy** for fatigue-associated state. Nothing in this repository should be read as clinical fatigue detection.

## Relationship to the Original Study

| Aspect | Ebied et al. (2020) | This repository |
|---|---|---|
| Dataset, subjects, acquisition, protocol | Defined and collected by the original authors | Used as provided; not collected by me |
| Feature definitions | RMS and MDF per 1 s epoch | Same two features, re-implemented in Python |
| Fatigue parameters | ΔRMS and ΔMDF = slope of a first-order fit to the 120 epoch values | Same definition, re-implemented |
| Statistics | Averages across channels/subjects; one-sample t-test of slopes against zero (uncorrected) | Same t-tests reproduced; additional analyses listed in the next section |
| Software | MATLAB | Python (NumPy, SciPy, pandas, scikit-learn) |
| Stated aims | Identify most/least fatigue-susceptible channels; classify subjects by endurance (reported as needing further work). JASA is suggested in the paper as future work | Channel/subject patterns re-examined; JASA-style zones, onset modelling and an exploratory ranking attempted (all unvalidated) |
| Machine learning | None | Entire classification pipeline is new to this repository |

### How my reproduction compares with the paper

The paper reports results as figures and short statements, not tables, so the comparison is qualitative.

| Observation | Ebied et al. | This implementation |
|---|---|---|
| ΔMDF sign | Mostly negative | 85.0% of 120 subject-channel series negative |
| ΔRMS sign | Mostly positive | 90.0% of series positive |
| Channel with largest MDF shift | Ch4 | Ch4 (mean −0.1128 Hz/s) |
| Channel with smallest MDF shift | Ch2 (only non-significant channel) | Ch2 (−0.0142 Hz/s); **Ch1 and Ch2** both non-significant at uncorrected α = 0.05 (Ch1: p = 0.070) |
| Channel without significant RMS change | Ch3 only | Ch3 only (p = 0.069) |
| Subjects with smallest MDF shift | Sub3 and Sub6 (both non-significant) | Sub6 and Sub4 have the smallest \|mean ΔMDF\|; Sub3 is significant at uncorrected α (p = 0.043) |
| Subjects without significant RMS change | Sub1, Sub8, Sub14 | Sub1, Sub7, Sub8, Sub14 (Sub7: p = 0.060) |

Differences to be aware of:

- The paper states that an sEMG **envelope** was computed before epoching for the amplitude analysis. The notebook computes RMS directly from the epoch-demeaned raw signal and does not implement an envelope step.
- Absolute ΔRMS magnitudes appear larger in the paper's figures than here (for example, the channel-mean for Ch7 reads as roughly 0.24 in the paper's Fig. 6D versus 0.17 here). The cause has not been established. 
- The paper does not give STFT window, overlap or interpolation details; the MDF implementation here (one Hann-windowed FFT per 1 s epoch, linear interpolation) is my own reading of the method.

## Extensions Implemented in This Project

Each item below exists in the notebook and is not part of the Ebied et al. paper in substantially the same form. The *methods* themselves (Holm correction, AICc, logistic regression, LOSO, etc.) are standard; what is new here is their application to this dataset.

| # | Extension | Notebook section | Note |
|---|---|---|---|
| 1 | Signal-quality summary: duration, range, clipping at 8-bit limits, mean offset, NaN and flat-channel checks, mean PSD | 1.3 | Paper reports no QC |
| 2 | Linear-fit goodness-of-fit (R²) for every subject-channel series | 3.1 | Paper reports slopes only |
| 3 | Wilcoxon signed-rank tests, Holm and Benjamini–Hochberg correction, 95% CIs, and an overall test on subject means | 3.2 | Paper used uncorrected one-sample t-tests only |
| 4 | Alternative spectral indices: notch-excluded MDF, log spectral-moment ratio (log SMR), log low/high power ratio (log L/H), compared against MDF and RMS | 4 | Paper uses MDF only.  |
| 5 | Fatigue-zone analysis (four zones from the signs of the RMS and spectral slopes) and an exploratory per-subject score | 5 | The paper only cites JASA as a suggested future direction.  |
| 6 | Trajectory modelling: linear vs. continuous piecewise-linear vs. saturating exponential, AICc-based selection, breakpoint-based onset estimate | 6 | Not in the paper |
| 7 | Exploratory composite fatigue-response ranking of subjects | 7 | Paper names this as an unresolved goal; the ranking here is unvalidated |
| 8 | Additional features for classification: MAV, WL, ZC (with RMS, MDF and the spectral indices above) | 8 | |
| 9 | Windowed dataset, per-subject baseline normalization, binary and three-class temporal-window labels | 9 | |
| 10 | Leave-one-subject-out evaluation of logistic regression, random forest and histogram gradient boosting; bootstrap CIs; paired Wilcoxon comparisons with Holm correction | 9–10, 15 | |
| 11 | Within-subject label-permutation test and exact subject-level sign-flip test | 10.4 | |
| 12 | Three-class (early/middle/late) task | 11 | |
| 13 | Sensitivity analysis over window length and label fraction | 12 | |
| 14 | Channel-contribution analysis (single channel, drop-one, in-fold top-k) | 13 | |
| 15 | Random-forest feature importance by feature type and channel | 14 | Exploratory; fitted on all subjects |

## Analysis Pipeline

```
All_15sub_Data.mat (15 subjects × ~24,050-24,190 samples × 8 channels, 200 Hz)
  → loading, truncation to 120 s, quality control
  → 1 s epochs (demeaned): RMS, MDF, MDF (notch-excluded), log SMR, log L/H, MAV, WL, ZC
  → linear slopes (ΔRMS, ΔMDF) and R² per subject × channel
  → slope statistics (t-test, Wilcoxon, Holm/BH, CIs)
  → spectral-index comparison
  → fatigue-zone analysis
  → trajectory models and onset estimation → exploratory subject ranking
  → 5 s windows (1 s step) → per-subject baseline normalization (first 10 s)
  → early/late (and early/middle/late) labels
  → LOSO classification (LR, RF, HistGB)
  → permutation and sign-flip tests, sensitivity, channel and feature-importance analyses
```

## Dataset and Attribution

The dataset used here originates from, and is associated with, the study:

> A. Ebied, A. M. Awadallah, M. A. Abbass, and Y. El-Sharkawy, "Upper Limb Muscle Fatigue Analysis Using Multi-channel Surface EMG," in *Proceedings of NILES2020: 2nd Novel Intelligent and Leading Emerging Sciences Conference*, IEEE, 2020, pp. 423–427.

I did not collect the data. All credit for the experimental design, acquisition and the original RMS/MDF slope methodology belongs to the original authors.

**Dataset characteristics (as described in the paper and confirmed against the notebook output):**

| Item | Value |
|---|---|
| Subjects | 15 healthy young male adults (age 21.56 ± 1.84 y; mass 73.55 ± 3.54 kg; height 177.92 ± 2.58 cm, per the paper) |
| Device | Myo armband, 8 stainless-steel single-differential sEMG channels, Ch4 over pronator teres (per the paper) |
| Sampling | 200 Hz, 8-bit signed; the paper states the armband output is already notch-filtered at 50 Hz |
| Protocol | Standing, right elbow at 90°, 6 kg load, forearm horizontal, palm up, recorded continuously for 120 s (per the paper) |
| Recording length in file | 24,049-24,193 samples per subject (about 120.2-121.0 s); truncated here to 24,000 samples = 120 epochs |
| Layout expected by the notebook | `All_15sub_Data.mat` containing a 15-element struct array `Sub`, each with an `emg` field of shape (samples, 8) |

**Source and license.** The dataset is hosted on Zenodo (https://zenodo.org/records/5189275, v1, DOI 10.5281/zenodo.5189275) under CC BY 4.0, which allows reuse and redistribution with attribution. This repository does not include the raw data (download instructions are in `data/README.md`). Do not commit the paper's PDF, which is copyrighted.

**Placing the data locally.** Download `Dataset EMG Fatigue.zip` from the Zenodo record, extract the `.mat` file with all subjects, then either:

```
<PROJECT_DIR>/All_15sub_Data.mat
```

or set `MAT_PATH` to the file's location. Results are written to `<PROJECT_DIR>/results`. The notebook's default file name is `All_15sub_Data.mat`; rename the extracted file if its name differs.

## Signal Processing and Feature Extraction

No band-pass or additional notch filtering is applied in the notebook. Each 1 s epoch (200 samples) is demeaned, then:

| Feature | Definition in this notebook |
|---|---|
| RMS | Root mean square of the demeaned epoch |
| MDF | Frequency splitting the epoch power spectrum into two equal-power halves (Hann-windowed FFT, 0 Hz to Nyquist, linearly interpolated) |
| MDF (notch-excluded) | MDF after removing the 45–55 Hz bins |
| log SMR | log10(M₋₁ / M₅), spectral moments for f ≥ 5 Hz |
| log L/H | log10(power in 10–45 Hz / power in 55–95 Hz) |
| MAV, WL, ZC | Mean absolute value; waveform length; zero crossings with a ≥ 1-unit amplitude dead band |

RMS and MDF follow Ebied et al.; the remaining features are additions (see Extensions).

**Quality control (all 15 subjects):** no NaNs and no flat channels. Samples at the 8-bit range limits (−128 / 127) range from about 0.24% (Sub3, Sub9) up to 3.49% (Sub4), with Sub10 at 3.17%. The mean PSD shows the expected dip at 50 Hz and further dips whose origin was not investigated. MDF values across the dataset span 2.0–91.1 Hz.

## Statistical and Temporal Analysis

- **Slopes.** A straight line is fitted to each of the 120 subject × channel series of RMS and MDF; the slope is ΔRMS or ΔMDF and R² measures how well a line describes the trajectory.
- **Tests.** Channel-level tests use the 15 subjects as observations; subject-level tests use the 8 channels (correlated, so descriptive only); the overall test uses one mean slope per subject. One-sample t-test and Wilcoxon signed-rank test against zero, Holm and Benjamini–Hochberg correction, 95% CIs.
- **Spectral-index comparison.** Each index is compared with RMS by the share of series with the expected slope sign, slope t statistics, channel-level tests and effect size.
- **Fatigue zones.** Each series is assigned to *Fatigue*, *Force increase*, *Recovery* or *Force decrease* from the signs of its RMS and MDF slopes. The fraction of a subject's channels in the Fatigue zone is an exploratory score. No independent fatigue measure exists to validate it.
- **Trajectory models.** Linear, continuous two-segment piecewise-linear (breakpoint searched from 10 s to 110 s) and saturating exponential models, compared by AICc (a non-linear model is preferred only if it improves AICc by ≥ 2). A breakpoint counts as a valid *onset* only if the piecewise model is preferred and the second segment moves further in the fatigue direction than the first.
- **Exploratory ranking.** Mean of three ranks: fatigue-zone score, median MDF onset and median RMS onset.

## Machine-Learning Pipeline

**Task.** Subject-independent early-versus-late task-state classification (and a three-class early/middle/late variant). Labels derive from time within the recording, which is a proxy for fatigue state and not an independently measured ground truth.

**Windows and labels.**

| Item | Setting |
|---|---|
| Window | Mean of the epoch features over 5 consecutive epochs, 1 s step |
| Binary labels | *Early*: window entirely within the first 25% (30 s); *Late*: window starts in the last 25% (from 90 s); intermediate windows excluded |
| Three-class labels | Window entirely within the first, second or third 40 s block |
| Baseline exclusion | Windows starting inside the first 10 s are excluded from every experiment |
| Binary dataset | 630 windows (240 early, 390 late), 42 per subject; majority-class accuracy 0.619 |
| Three-class dataset | 1,470 windows (390 / 540 / 540) |

**Feature sets (8 channels each):** Core (RMS, MDF; 16 features), Spectral-index (RMS, MDF notch-excluded, log SMR, log L/H; 32), Combined (all eight features; 64). A single-feature reference uses channel-averaged MDF with logistic regression.

**Baseline normalization.** Each feature is expressed relative to the same subject's mean over the first 10 s (relative change; subtraction for log-scale features). It uses only that subject's own initial segment, but it **requires a 10 s calibration recording for every subject**, including a new one. Raw (unnormalized) features were evaluated on identical windows.

**Models (fixed hyperparameters, not tuned).** Logistic regression (standardization + C = 1), random forest (200 trees, min leaf 2), and scikit-learn's histogram-based gradient boosting (depth 3, learning rate 0.1, 150 iterations; reported as "Gradient Boosting").

**Evaluation.** Leave-one-subject-out (LOSO): each subject is held out in turn; standardization and model fitting use the other 14 subjects only. Subject-level results (n = 15) are the unit of inference, summarized with a bootstrap CI over subjects; windows overlap and are not independent. Metrics: accuracy, balanced accuracy, F1, AUC (per held-out subject, then averaged).

**Statistical validation.** Within-subject label permutation test (1,000 permutations; smallest attainable p = 0.001) and exact subject-level sign-flip test, both on the configuration fixed a priori (Combined features, logistic regression, baseline-normalized). Paired Wilcoxon comparisons between configurations with Holm correction.

## Key Results

### Physiological / signal-analysis findings

| Quantity | Result |
|---|---|
| Mean MDF, first 10 s → last 10 s | 63.4 Hz → 56.3 Hz |
| Overall ΔMDF (mean of subject means) | −0.0590 Hz/s, 95% CI [−0.0765, −0.0416], p = 4.13e-6 |
| Overall ΔRMS | +0.1010 RMS units/s, 95% CI [0.0709, 0.1312], p = 4.68e-6 |
| Series with expected slope sign | ΔRMS > 0: 90.0%; ΔMDF < 0: 85.0% |
| Median linear-fit R² | RMS 0.453; **MDF 0.087** |
| Series with R² < 0.1 | RMS 14.2%; **MDF 54.2%** |
| Channels significant after Holm (n = 15 subjects) | ΔMDF: 6/8 (not Ch1, Ch2); ΔRMS: 7/8 (not Ch3) |
| Subjects significant after Holm (across 8 channels; descriptive) | ΔMDF: 2/15 (Sub10, Sub13); ΔRMS: 6/15 (Sub3, 4, 9, 10, 11, 13) |
| Fatigue zone, MDF-based | Fatigue 77.5%, Force increase 12.5%, Recovery 2.5%, Force decrease 7.5% of series |
| Fatigue-zone share by channel | Lowest Ch1 (0.60), highest Ch6 (0.93) |
| Fatigue-zone ranking vs. mean-ΔMDF ranking | Spearman ρ = 0.42, p = 0.118 (n = 15; not independent rankings) |

**Spectral-index comparison (expected direction, per series):**

| Index | % expected sign | % significant in expected direction | Median R² | Channels sig. (Holm) |
|---|---|---|---|---|
| RMS | 90.0 | 86.7 | 0.453 | 7 |
| MDF | 85.0 | 60.0 | 0.087 | 6 |
| MDF (notch-excluded) | 70.8 | 50.8 | 0.065 | 3 |
| log SMR | 72.5 | 52.5 | 0.072 | 3 |
| log L/H | 87.5 | 68.3 | 0.104 | 6 |

**Trajectory models:**

| Signal | Piecewise beats linear | Exponential beats linear | Median R² (lin / piecewise / exp) | Valid onset | Median onset (IQR) |
|---|---|---|---|---|---|
| MDF | 37.5% | 15.0% | 0.09 / 0.14 / 0.11 | 27.5% | 56.0 s (43.0–79.0) |
| RMS | 73.3% | 24.2% | 0.45 / 0.55 / 0.50 | 38.3% | 74.0 s (39.8–95.5) |

### Classification results (LOSO, mean over 15 held-out subjects)

Binary early-vs-late, baseline-normalized:

| Feature set | Model | Accuracy [95% CI] | Balanced acc. | AUC |
|---|---|---|---|---|
| Spectral-index (32) | Random Forest | 0.956 [0.913, 0.989] | **0.954** (highest) | 0.993 |
| Combined (64) | Logistic Regression | 0.921 [0.849, 0.979] | 0.929 (pre-specified, permutation-tested) | 0.973 |
| Core (16) | Logistic Regression | 0.875 [0.790, 0.948] | 0.886 | 0.954 |
| MDF-only reference (1 feature) | Logistic Regression | 0.795 [0.716, 0.876] | 0.784 | 0.930 |
| Majority-class rate | n/a | 0.619 | 0.500 | 0.5 |

All nine normalized model/feature-set combinations reach balanced accuracy 0.886–0.954.

**Validation of the pre-specified configuration** (Combined features, logistic regression, baseline-normalized): observed balanced accuracy 0.929; permutation null mean 0.454, 95th percentile 0.536; permutation p = 0.001 (the minimum attainable with 1,000 permutations); sign-flip p = 0.00003. The Spectral-index random forest is the highest-scoring configuration in the table but was chosen after seeing results among several configurations, so its figure is optimistically selected and the permutation test was not run on it.

**Other findings:**

| Question | Result |
|---|---|
| Effect of baseline normalization (Core features, accuracy) | Raw → normalized: +0.241 (LR), +0.216 (RF), +0.217 (GB); Holm-adjusted p = 0.024, 0.024, 0.010; improved in 12, 12 and 14 of 15 subjects |
| Best raw-feature configuration | Random Forest, Spectral-index: balanced accuracy 0.739, accuracy 0.752 (Gradient Boosting has the highest raw accuracy, 0.756; raw AUCs are 0.94–0.995, i.e. ordering within a subject is preserved but a shared threshold transfers poorly) |
| Richer features vs. Core (Spectral-index or Combined) | No difference after Holm correction (all adjusted p ≥ 0.63) |
| Core vs. MDF-only reference | +0.079 (LR), +0.146 (RF), +0.140 (GB) accuracy; Holm-adjusted p = 0.79, 0.078, 0.125, so not conclusive |
| Three-class, best by macro F1 | Spectral-index, Random Forest: accuracy 0.684, balanced accuracy 0.693, macro F1 0.667 (chance ≈ 0.33); middle class is confused with both neighbours; early→late and late→early confusions are 0.00 |
| Sensitivity (LR, balanced acc.) | Combined: 0.891–0.947 across window 3/5/10 s × label fraction 0.20/0.25/0.33; Core: 0.870–0.956; performance is lower for the 0.33 fraction |
| Channel contribution (Combined, LR; all-channel acc. 0.921) | Single-channel accuracy highest for Ch4 (0.886), Ch5 (0.876), Ch6 (0.857), lowest Ch3 (0.673). Dropping Ch4 or Ch5 lowers accuracy by 0.044 / 0.038; dropping Ch8 *raises* it by 0.022. In-fold top-k: 0.816 (k=1), 0.929 (k=3), 0.944 (k=4), 0.921 (k=8) |
| RF importance (fit on all subjects) | By feature type: MAV and RMS largest, then ZC and log L/H; MDF smaller. By channel: Ch5 and Ch4 largest. Single largest: ZC Ch4 (0.105) |

## Selected Visualizations

The notebook exports these eight figures to `results/figures/` (Section 17); the other figures are shown in the notebook only. Run the notebook on the dataset to generate them.

![Mean ΔMDF and ΔRMS per subject and channel with 95% CIs](../results/figures/mean_slopes_ci.png)

![Linear-fit R² distribution for RMS and MDF](../results/figures/linear_fit_r2_histograms.png)

![RMS, MDF and alternative spectral indices compared](../results/figures/spectral_index_comparison.png)

![Fatigue-zone assignment per subject-channel series](../results/figures/fatigue_zone_analysis.png)

![Example trajectory model comparison, Subject 1 Channel 4](../results/figures/trajectory_example_fits.png)

![Binary early-vs-late LOSO accuracy by feature set, model and normalization](../results/figures/binary_accuracy_by_feature_set.png)

![Three-class temporal-window confusion matrix](../results/figures/three_class_confusion_matrix.png)

![Channel contribution analysis](../results/figures/channel_contribution_analysis.png)

## Interpretation

- On average, MDF decreased and RMS increased over the 120 s hold. This direction matches what is typically described for sustained contractions and matches the original paper's observations, and Ch4 was the most MDF-affected channel in both.
- The MDF trend is real on average but weak within individual series: a straight line explains little variance (median R² 0.087) and more than half of the series have R² < 0.1. Per-subject MDF significance largely disappears after Holm correction. RMS trends were more consistent. Between-subject and between-channel heterogeneity is substantial (for example Ch1/Ch2 for MDF, Ch3 for RMS).
- Non-linear models fit somewhat better, especially for RMS, but the improvement for MDF is modest and onset estimates exist for only 27.5% (MDF) and 38.3% (RMS) of series. Onset times and the subject ranking should be viewed as descriptive.
- Early and late segments of the recording are separable in held-out subjects when features are expressed relative to each subject's own first 10 s. Without that baseline, accuracy falls to 0.63–0.76 even though AUC stays high. The result supports "feature patterns change systematically over the task and transfer across these subjects after per-subject calibration". It does not show that the classifier measures fatigue specifically: any monotonic change over time (electrode/skin changes, posture, force drift) could contribute.
- The binary task compares early (10–30 s) with late (90–120 s) windows, which are far apart in time. The harder three-class task gives balanced accuracy of 0.55–0.69.

## Limitations

- **Small, homogeneous cohort:** 15 healthy young males, one task, one device; subject-level confidence intervals are wide and tests descriptive. No external-dataset validation.
- **Proxy labels:** no independent fatigue ground truth (force, MVC, time to failure, perceived exertion); temporal position is the label.
- **Calibration requirement:** best results depend on a 10 s per-subject baseline; windows overlap and are not independent.
- **Signal limits:** 8-bit data at 200 Hz (Nyquist 100 Hz), with a hardware 50 Hz notch that falls inside the MDF range; 0.24–3.49% of samples sit at the amplitude limits, and the mean spectrum has additional unexplained dips.
- **Simple linear modelling:** per-series linear slopes describe MDF poorly; p-values from within-series fits assume independent epochs; subject-level tests across channels treat correlated channels as observations.
- **Selection effects:** hyperparameters and settings were fixed in advance, but the highest-scoring configuration was identified from the same LOSO results. RF importance was fitted on all subjects.
- **Reproduction gaps:** no envelope step; possible differences from the original MATLAB/STFT implementation (see above).
- **Exploratory outputs:** the fatigue-zone score, composite ranking and onset times are unvalidated.

## Repository Structure

Layout of the published repository:

```
.
├── EMG_Fatigue_Analysis.ipynb   
├── README.md                          
├── docs/METHODS.md                    
├── requirements.txt                   
├── LICENSE                            
├── data/                              
│   └── README.md                      
└── results/                           
    ├── figures/                       
    ├── tables/                        
    └── supplementary/                 
```

`the dataset is downloaded from Zenodo.

## Installation

Dependencies are taken from the notebook's imports (also listed in `requirements.txt`). Standard-library imports (`itertools`, `os`, `re`, `warnings`) need no installation.

| Package | Used for |
|---|---|
| numpy, scipy, pandas | arrays, signal/statistics, tables |
| matplotlib | figures (including 3-D bar charts) |
| scikit-learn | models, metrics, LOSO |
| jupyter / ipython | notebook execution and `display` |
| h5py | only if the `.mat` file is MATLAB v7.3 (HDF5) |

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python and package versions: `[TO CONFIRM]`. Record them with `pip freeze` once the notebook runs end to end.

## Usage

1. Place `All_15sub_Data.mat` in `data/` (see the dataset section) or set the `MAT_PATH` environment variable. The notebook also accepts the file in the project root.
2. Optionally set `PROJECT_DIR` (default: current directory). Key figures go to `<PROJECT_DIR>/results/figures`, key tables and the headline summary to `<PROJECT_DIR>/results/tables`, and complete intermediate tables to `<PROJECT_DIR>/results/supplementary`. Pointing `PROJECT_DIR` at a Google Drive (or other synced) folder stores the results there.
3. Launch Jupyter and run `EMG_Fatigue_Analysis_Clean.ipynb` from top to bottom. The permutation test (Section 10.4, 1,000 permutations) is the slowest step.

The notebook contains no machine-specific or Colab paths.

## Future Work

- Validate the temporal-state classifier against recordings with an independent fatigue measure (force decline, MVC loss or time to failure).
- Add the envelope step described in the original paper and compare it, and alternative STFT/Welch settings, against the current RMS and MDF values.
- Test how short the per-subject baseline can be before LOSO performance drops, and add a gap between early and late windows or use non-overlapping windows.
- Model subject-channel nesting directly (for example mixed-effects models) instead of testing channels or subjects separately.
- Investigate the clipped samples and the extra spectral dips, and check whether results change when affected epochs are excluded.
- Test on another subject cohort or dataset.

## Citation

If you reuse this repository, please cite and credit the original study and dataset:

```bibtex
@inproceedings{ebied2020upper,
  author    = {Ebied, Ahmed and Awadallah, Ahmed M. and Abbass, Mohamed A. and El-Sharkawy, Yasser},
  title     = {Upper Limb Muscle Fatigue Analysis Using Multi-channel Surface {EMG}},
  booktitle = {Proceedings of NILES2020: 2nd Novel Intelligent and Leading Emerging Sciences Conference},
  publisher = {IEEE},
  year      = {2020},
  pages     = {423--427},
  doi       = {10.1109/NILES50944.2020.9257909}
}
```

Dataset citation: A. Ebied, A. M. Awadallah, M. A. Abbass, and Y. El-Sharkawy, "Multi-channel Surface EMG Dataset for Fatigue analysis," Zenodo, v1, 2021, doi:10.5281/zenodo.5189275.

Acknowledgement text: *"This work uses the multichannel sEMG fatigue dataset associated with Ebied et al. (2020) and builds on the RMS/MDF slope methodology described in that paper."* To cite this repository, use the "Cite this repository" button on GitHub (generated from `CITATION.cff`).

## License

The code is released under the MIT License (see `LICENSE`). The dataset is licensed separately by its authors under CC BY 4.0, and the code license does not apply to the dataset or to the original paper.
