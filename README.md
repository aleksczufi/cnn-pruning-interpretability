# Transferring Visual Explanations: Knowledge Distillation and Interpretability

Code for the paper *"Transferring Visual Explanations: How Cross-Architecture Knowledge Distillation Affects Model Interpretability"* (A. Czufarow, I. Babin).

The paper studies whether knowledge distillation (KD) transfers a teacher's spatial attribution to the student, measured with Grad-CAM and Guided Grad-CAM heatmaps scored by Relevance Mass Accuracy (RMA) and Relevance Rank Accuracy (RRA) against VOC2012 annotations.

## Contents

- `pruning-for-vision-representation/` – distillation driver (`distill.py`), data loading, Grad-CAM / Guided Grad-CAM evaluation and RMA/RRA metrics, launch scripts.
- `data/voc2012_eval_images_table1.txt` – the 30 VOC2012 images used for the cross-architecture experiments (Table 1).
- `data/voc2012_eval_images_table2.txt` – the 30 VOC2012 images used for the temperature / soft-label weight sweep (Table 2).
- `data/voc2012_figure_examples.txt` – images shown in Figures 2–4 (not part of the evaluation samples).
- `requirements.txt` – library versions used in the experiments.
- `train.py`, `results/` – scripts and outputs from an earlier pruning stage of the project; not used for the results reported in the paper.

## Setup

```
pip install -r requirements.txt
```

Datasets (not included): ImageNet-1K (ILSVRC-2012) and PASCAL VOC2012, available from their official websites.

## Data and checkpoints

Per-epoch metric logs and trained checkpoints are available from the corresponding author on reasonable request.
