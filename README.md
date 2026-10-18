# ChestInsight — MATLAB-only chest X-ray project

This is a MATLAB-only academic prototype for multi-label analysis of frontal
chest X-rays. It evaluates ten findings independently:

`Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`,
`Pneumonia`, `Pneumothorax`, `Consolidation`, and `Edema`.

The project does **not** need Python, PyTorch, Streamlit, a web browser, or a
GPU. It runs in MATLAB R2026a using base MATLAB functions. The code does not
need the Deep Learning Toolbox or Image Processing Toolbox.

## Run the application

1. Double-click `Start MATLAB App.cmd`.
2. Click **Use built-in example** or **1. Choose X-ray**.
3. Select the finding you want to inspect.
4. Click **2. Analyze in MATLAB**.
5. Read the ten scores and view the MATLAB contrast-enhanced image and saliency
   map. A PNG, MAT file, JSON, CSV, and images are automatically saved under
   `reports/matlab_only_runs`.

Use de-identified frontal AP/PA PNG or JPEG chest X-rays only.

## Retrain the MATLAB model

Double-click `Train MATLAB Model.cmd`.

It builds MATLAB features from `data/images`, trains a two-hidden-layer ReLU
neural network, selects each finding threshold using validation patients, and
evaluates once on untouched test patients. The resulting model is saved at:

`models/chestinsight_matlab_model.mat`

The first run creates `data/matlab_features.mat`; later runs reuse it. Delete
that file only when the dataset images or manifest are changed.

To train from the MATLAB Command Window:

```matlab
cd('C:\Users\Mathews\Documents\ChatGPT\BME')
addpath('matlab')
chestinsight_train('MaxEpochs',10)
chestinsight_app
```

## How it works

1. MATLAB converts the X-ray to grayscale, normalizes it, and resizes it to
   32×32 pixels.
2. A custom two-hidden-layer neural network produces ten independent sigmoid
   scores. Several findings can be above threshold at the same time.
3. MATLAB calculates an input-gradient saliency map for the selected finding.
   It shows which image pixels most affect that output; it is not a lesion
   boundary.
4. MATLAB creates a display-only contrast-enhanced view, histogram, result
   table, PNG, CSV, MAT file, and JSON summary.

## Measured MATLAB-only pilot result

The included model was trained on 4,312 images from 1,306 patients, selected
with 1,096 images from 327 validation patients, and evaluated on 2,560 images
from 162 untouched test patients.

| Metric | Result |
| --- | ---: |
| Macro AUROC | 0.593 |
| Macro average precision | 0.117 |
| Macro F1 | 0.168 |

See `reports/matlab_only/test_metrics.csv` for results for every finding.

## Limits

This is an academic research prototype. It uses a small, non-random pilot
subset of NIH ChestX-ray14 and report-derived labels. Scores are not calibrated
clinical probabilities, and the system is not validated for diagnosis or
treatment.
