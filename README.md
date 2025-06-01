
# Face Attribute Classifier

This project implements a multi-task facial attribute classifier developed as part of a job interview assignment. The model performs binary classification on facial images while simultaneously predicting auxiliary attributes (hat, glasses, facial hair) to improve feature representation and classification performance.

---

## Problem Overview

The dataset consists of face images labeled with a binary classification:

* `1` — Clear face
* `0` — Face with one or more of the following attributes: hat, glasses, or facial hair

The `0` class is inherently ambiguous, encompassing multiple distinct visual features, which introduces challenges in classification.

---

## Label Expansion Using BLIP VQA

To resolve ambiguity in the `0` class, the BLIP Visual Question Answering (VQA) model was used to automatically annotate each image with three binary facial attributes:

* `hat` — 1 if the subject is wearing a hat, otherwise 0
* `glasses` — 1 if the subject is wearing glasses, otherwise 0
* `facial_hair` — 1 if the subject has facial hair, otherwise 0

This resulted in a refined labeled dataset with the following format:

```
image_path,hat,glasses,facial_hair,label
```

---

## Multi-Task Model Architecture

The model employs a shared ResNet encoder with four classification heads:

* A primary head for the original binary label classification
* Three auxiliary heads, each responsible for predicting one of the facial attributes: hat, glasses, and facial hair

### Training Strategy

* A weighted multi-task loss function is used, with configurable weights for the main label task (`label_weight`) and auxiliary tasks (`aux_weight`).
* Class weights can optionally be applied to address label imbalance in the main classification task.

---

## Evaluation and Inference

The evaluation process includes threshold optimization for each classification head and generates final predictions on the provided test set, which does not include ground truth labels. The output predictions are saved in `test_labels.txt`.

---

## Project Structure

```
.
├── label.ipynb           # Label expansion using BLIP VQA for attribute annotation
├── train.py              # Multi-task training script
├── eval.ipynb            # Threshold tuning and test set inference
├── utils.py              # Utility functions for dataset handling and evaluation
├── labeled_data.csv      # Final labeled training data with binary attributes
├── label_train.txt       # iInitial labels for the training set
├── label_val.txt         # Predictions on the provided test set
├── labeled_data.csv      # Final labeled training data with binary attributes
# Final labeled training data with binary attributes
├── checkpoints/
│   ├── train_split.csv   # Training set split used for model development
│   ├── val_split.csv     # Validation set split used for model development
│   ├── test_df.csv       # Metadata for the test set
│   └── test_labels.txt   # Final predictions for the test set
```

> **Note:** Model checkpoints (trained weights) are not included in this repository due to file size constraints.

---

## Training Instructions

The model can be trained using the following command:

```bash
python train.py \
  --csv_path labeled_data.csv \
  --root_dir path/to/images \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0001 \
  --weight_decay 1e-4 \
  --save_dir checkpoints \
  --device cuda \
  --num_workers 4 \
  --use_class_weights \
  --label_weight 0.7 \
  --aux_weight 0.1
```

### Argument Descriptions:

* `--csv_path`: Path to the CSV file containing image paths and labels
* `--root_dir`: Root directory of the image dataset
* `--epochs`: Number of training epochs (default: 50)
* `--batch_size`: Batch size for training (default: 32)
* `--lr`: Learning rate (default: 0.0001)
* `--weight_decay`: Weight decay factor (default: 1e-4)
* `--save_dir`: Directory to save checkpoints and outputs (default: `checkpoints`)
* `--device`: Device to use for training (`cpu`, `cuda`, or `auto`)
* `--num_workers`: Number of worker threads for data loading (default: 4)
* `--use_class_weights`: Flag to enable class weighting for the main label task
* `--label_weight`: Weight of the main label loss in the multi-task loss function (default: 0.7)
* `--aux_weight`: Weight of each auxiliary task loss (default: 0.1)

