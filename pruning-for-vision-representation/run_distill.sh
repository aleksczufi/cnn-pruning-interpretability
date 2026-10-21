#!/bin/bash
# Runner for distill.py (knowledge distillation)
# Usage: ./run_distill.sh [DATA_PATH]
#
# Fine-tuning + augmentation recipe: pretrained ResNet-18 distilled from ResNet-50.
# Plain KD on the same data the student was already trained on gives no new signal.
# Mild auto-augment makes training views harder, so the teacher's soft labels
# help the student learn robust features beyond its pretrained baseline.
#
# Recipe (standard 256/224 eval is unchanged):
#  - train-crop 176, eval 224: FixRes effect (torchvision V2 recipe) boosts 224-eval
#    accuracy AND cuts per-epoch compute ~1.6x, so more epochs fit in the budget.
#  - wd 1e-4 (not 5e-4): high WD x LR is what dragged the pretrained weights away
#    from their optimum in the lr=0.01 run (69.8% -> 63.9%, still falling).
#  - kd-alpha 0.9, no label smoothing: mostly-distillation objective; smoothed
#    hard labels conflict with the teacher's soft targets.
#  - lr 0.005 cosine over 30 epochs with 3-epoch warmup. NOTE: the scheduler
#    steps per-epoch, so --lr-warmup-epochs N is N ramp steps; keep >=3.
# Throughput: ~20 min/epoch when this is the ONLY job on the GPU (~2h/epoch when
# sharing with another training run) -> 30 epochs ~ 12h solo.

set -e

cd "$(dirname "$0")"

DATA_PATH="${1:-/mnt/data/ImageNet/ILSVRC/Data/CLS-LOC}"
OUTPUT_DIR="./results_kd/igor"

python distill.py \
    --teacher resnet50 \
    --student resnet18 \
    --student-pretrained \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 512 \
    --workers 16 \
    --epochs 30 \
    --lr 0.005 \
    --lr-warmup-epochs 3 \
    --lr-warmup-method linear \
    --opt sgd \
    --momentum 0.9 \
    --wd 1e-4 \
    --train-crop-size 176 \
    --auto-augment imagenet \
    --kd-temperature 4.0 \
    --kd-alpha 0.9 \
    --amp \
    --seed 1
