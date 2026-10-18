# MATLAB-only delivery

## Files to open

- `Start MATLAB App.cmd` — opens the user interface.
- `Train MATLAB Model.cmd` — retrains the MATLAB model and refreshes metrics.
- `reports/matlab_only_verified/matlab_only_report.png` — verified sample output.
- `reports/matlab_only/test_metrics.csv` — measured pilot test metrics.

## Simple demo flow

1. Start the MATLAB application.
2. Click **Use built-in example**.
3. Keep `Cardiomegaly` selected and click **Analyze in MATLAB**.
4. Explain that the bar chart shows ten independent findings, the middle image
   is contrast-enhanced for viewing only, and the colored image is a saliency
   map for the selected output.
5. Open the created folder in `reports/matlab_only_runs` to show the exported
   PNG, CSV, JSON, MAT file, enhanced image, and saliency image.

## What to say in the presentation

“Our system is a MATLAB-only multi-label neural-network prototype. One chest
X-ray can have more than one finding, so it produces ten independent scores.
MATLAB handles preprocessing, model training, inference, saliency, image
enhancement, evaluation, and report generation. We separated patients among
training, validation, and test data to avoid patient leakage.”

The included pilot model achieved macro AUROC 0.593 on 2,560 test images from
162 held-out patients. It is a research demonstration, not a clinical diagnosis
system.
