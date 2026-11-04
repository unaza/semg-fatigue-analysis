| Section | Metric | Value |
|---|---|---|
| Dataset | Subjects / epochs per subject / channels | 15 / 120 (1 s) / 8 |
| Signal analysis | Mean MDF, first 10 s -> last 10 s (Hz) | 63.4 -> 56.3 |
| Signal analysis | Overall ΔMDF, mean of subject means (Hz/s) | -0.0590 (95% CI -0.0765 to -0.0416), p = 4.13e-06 |
| Signal analysis | Overall ΔRMS, mean of subject means (RMS units/s) | 0.1010 (95% CI 0.0709 to 0.1312), p = 4.68e-06 |
| Signal analysis | Series with ΔRMS > 0 / ΔMDF < 0 (%) | 90.0 / 85.0 |
| Signal analysis | Median linear-fit R²: RMS / MDF | 0.453 / 0.087 |
| Signal analysis | Series with R² < 0.1 (%): RMS / MDF | 14.2 / 54.2 |
| Signal analysis | Channels significant after Holm (of 8): ΔMDF / ΔRMS | 6 / 7 |
| Signal analysis | Subjects significant after Holm (of 15, descriptive): ΔMDF / ΔRMS | 2 / 6 |
| Signal analysis | MDF-based zone share of series in the Fatigue zone (%) | 77.5 |
| Signal analysis | Series where a piecewise model beats linear (%): MDF / RMS | 37.5 / 73.3 |
| Signal analysis | Series with a valid onset (%): MDF / RMS | 27.5 / 38.3 |
| Classification (binary, LOSO) | Windows / majority-class accuracy | 630 / 0.619 |
| Classification (binary, LOSO) | Highest balanced accuracy, baseline-normalized (selected after seeing results) | Spectral-index + Random Forest: balanced acc. 0.954, accuracy 0.956 (95% CI 0.913 to 0.989), AUC 0.993 |
| Classification (binary, LOSO) | Pre-specified configuration (permutation-tested) | Combined + Logistic Regression, baseline-normalized: balanced acc. 0.929; null mean 0.454, null 95th pct 0.536; permutation p = 0.0010; sign-flip p = 0.00003 |
| Classification (binary, LOSO) | Highest balanced accuracy, raw features | Spectral-index + Random Forest: balanced acc. 0.739, accuracy 0.752 |
| Classification (binary, LOSO) | MDF-only reference, baseline-normalized | balanced acc. 0.784, accuracy 0.795 |
| Classification (binary, LOSO) | Normalization effect on accuracy, Logistic Regression: normalized vs raw (Core) | +0.241 (Holm p = 0.0242) |
| Classification (binary, LOSO) | Normalization effect on accuracy, Random Forest: normalized vs raw (Core) | +0.216 (Holm p = 0.0242) |
| Classification (binary, LOSO) | Normalization effect on accuracy, Gradient Boosting: normalized vs raw (Core) | +0.217 (Holm p = 0.0096) |
| Classification (three-class, LOSO) | Highest macro F1 configuration | Spectral-index + Random Forest: balanced acc. 0.693, macro F1 0.667, accuracy 0.684 |
