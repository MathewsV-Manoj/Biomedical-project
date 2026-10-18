---
license: other
license_name: nih-chest-xray-14
license_link: https://nihcc.app.box.com/v/ChestXray-NIHCC
size_categories:
- 100K<n<1M
task_categories:
- image-classification
pretty_name: NIH ChestX-ray14
tags:
- multi-label
- medical
- chest-x-ray
- radiology
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: image
    dtype: image
  - name: labels
    list:
      class_label:
        names:
          '0': Atelectasis
          '1': Cardiomegaly
          '2': Consolidation
          '3': Edema
          '4': Effusion
          '5': Emphysema
          '6': Fibrosis
          '7': Hernia
          '8': Infiltration
          '9': Mass
          '10': Nodule
          '11': Pleural_Thickening
          '12': Pneumonia
          '13': Pneumothorax
  - name: label_names
    list: string
  - name: image_id
    dtype: string
  - name: fold
    dtype: int8
  - name: patient_id
    dtype: int32
  - name: follow_up
    dtype: int32
  - name: patient_age
    dtype: int32
  - name: patient_sex
    dtype: string
  - name: view_position
    dtype: string
  - name: original_width
    dtype: int32
  - name: original_height
    dtype: int32
  - name: pixel_spacing_x
    dtype: float64
  - name: pixel_spacing_y
    dtype: float64
  - name: bboxes
    list:
    - name: bbox
      list: float32
      length: 4
    - name: label
      dtype:
        class_label:
          names:
            '0': Atelectasis
            '1': Cardiomegaly
            '2': Consolidation
            '3': Edema
            '4': Effusion
            '5': Emphysema
            '6': Fibrosis
            '7': Hernia
            '8': Infiltration
            '9': Mass
            '10': Nodule
            '11': Pleural_Thickening
            '12': Pneumonia
            '13': Pneumothorax
  splits:
  - name: train
    num_examples: 86524
  - name: test
    num_examples: 25596
---

## Description
The NIH ChestX-ray14 dataset (NIH Clinical Center), an extension of ChestX-ray8 from the CVPR 2017 paper. It has 112,120 frontal-view chest X-rays of 30,805 unique patients. Each image is labelled with any of 14 thoracic findings, text-mined from the associated radiology reports with NLP.

'
ChestX-ray dataset comprises 112,120 frontal-view X-ray images of 30,805 unique patients with the text-mined fourteen disease image labels (where each image can have multi-labels), mined from the associated radiological reports using natural language processing. Fourteen common thoracic pathologies include Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_thickening, Cardiomegaly, Nodule, Mass and Hernia, which is an extension of the 8 common disease patterns listed in our CVPR 2017 paper.
'

Labels (14): `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, `Pneumothorax`.

Images labelled `No Finding` in the source (about 54% of images) have an empty `labels` list. Per the NIH FAQ, 'No Finding' is not the same as 'normal': those images may show patterns outside the 14 categories, or uncertain findings.

The labels are NLP-extracted and noisy (NIH estimates >90% accuracy). The test labels come from the same NLP process and are not radiologist-verified. Improved label sets are linked from the original download site (e.g. MAPLEZ, NIH-CXR-LT).

The splits are the official patient-level `train_val_list.txt` (86,524 images, 28,008 patients) and `test_list.txt` (25,596 images, 2,797 patients). No patient appears in both. There is no official validation split. The `train` split has a `fold` column (0-9) that I assigned with multi-label iterative stratification (Sechidis et al., 2011, seed 42), **grouped by patient**: every image of a patient is in the same fold. Use a fold as validation (e.g. `fold == 0`, ~10%) without leaking patients between train and validation. Findings and 'No Finding' are balanced to within 1% across folds; Hernia, the rarest, has 14-15 per fold.

Images are the original 1024x1024 8-bit PNGs re-encoded as 8-bit grayscale (`L` mode) JPEG at quality 95. This halves the size relative to PNG (mean PSNR ~48.4 dB, SSIM ~0.988 vs. the PNGs), and all 256 gray levels are preserved. About 1% of the source PNGs are stored as RGBA with identical R/G/B channels and opaque alpha; these were converted to `L` losslessly before encoding.

- Website: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Paper: https://arxiv.org/abs/1705.02315

## Dataset Structure

- `image` - JPEG image, 1024x1024, 8-bit grayscale.
- `labels` - sequence of `ClassLabel` indices of the positive findings (empty for 'No Finding').
- `label_names` - names of the positive findings.
- `image_id` - original image filename stem (e.g. `00000001_000`).
- `fold` - patient-grouped stratified fold (0-9) for `train`, null for `test`.
- `patient_id`, `follow_up`, `patient_age`, `patient_sex`, `view_position` (`PA` / `AP`) - from `Data_Entry_2017_v2020.csv` (Patient ID, Follow-up #, Patient Age, Patient Sex, View Position). Per NIH, the image name suffix does not necessarily match the follow-up number.
- `original_width`, `original_height`, `pixel_spacing_x`, `pixel_spacing_y` - original image size and pixel spacing before the 1024x1024 resize, from the same file.
- `bboxes` - list of `{bbox: [x, y, w, h], label}` from `BBox_List_2017.csv`, in pixels on the 1024x1024 image, (x, y) = top-left corner. Present for 880 `test` images (984 boxes, 8 findings), empty otherwise. The source's `Infiltrate` is mapped to `Infiltration`.

## Usage
This dataset follows the same multi-label format as [timm/plant-pathology-2021](https://huggingface.co/datasets/timm/plant-pathology-2021) and [timm/amazon-from-space](https://huggingface.co/datasets/timm/amazon-from-space) for use with `timm` 1.0.31 or greater. The images are grayscale; timm converts them to 3-channel RGB by default.

```bash
python train.py --dataset hfds/timm/nih-chest-xray-14 --train-split train --val-split test \
  --task multilabel --target-key labels --num-classes 14 \
  --model resnet50.ram_in1k --pretrained --img-size 512 --batch-size 32 --epochs 30 \
  --opt adamw --lr 1e-4 --weight-decay 0.01 --amp
```

To hold out a patient-level validation fold instead of validating on `test`:

```python
from datasets import load_dataset
ds = load_dataset('timm/nih-chest-xray-14', split='train')
train_ds = ds.filter(lambda f: f != 0, input_columns='fold')
val_ds = ds.filter(lambda f: f == 0, input_columns='fold')
```

The standard metric is per-finding ROC AUC, averaged over the 14 findings.

## License / Terms
From the NIH FAQ: "The usage of the data set is unrestricted. But you should provide the link to our original download site, acknowledge the NIH Clinical Center and provide a citation to our CVPR 2017 paper."

Original download site: https://nihcc.app.box.com/v/ChestXray-NIHCC. Data provided by the NIH Clinical Center.

## Citation
```
@InProceedings{wang2017chestxray,
  author = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald},
  title = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3462--3471},
  year = {2017}
}
```
