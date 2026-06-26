import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

import numpy as np

torch.set_float32_matmul_precision('medium')

# ─────────────────────────────────────────────
# CIFAR-100 class names 
# ─────────────────────────────────────────────
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


# ─────────────────────────────────────────────
# TensorBoard helper: log a grid of images
# with predicted vs. true labels
# ─────────────────────────────────────────────
def log_prediction_grid(writer, model, dataset, device, tag="Predictions", n=16):
    """
    Picks n random images from `dataset`, runs the model,
    and writes a labelled image grid to TensorBoard.
    """
    model.eval()
    indices = torch.randperm(len(dataset))[:n]
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = [dataset[i][1] for i in indices]

    with torch.no_grad():
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().tolist()

    # Build per-image title strings  ✓ correct  ✗ wrong
    titles = []
    for pred, true in zip(preds, labels):
        marker = "✓" if pred == true else "✗"
        titles.append(f"{marker} P:{CIFAR100_CLASSES[pred]}\nT:{CIFAR100_CLASSES[true]}")

    # Denormalise (images are plain ToTensor → already [0,1])
    grid = torchvision.utils.make_grid(images.cpu(), nrow=4, normalize=True)
    writer.add_image(tag, grid)

    # Also write the label strings as text so they are searchable
    writer.add_text(tag + "/labels", "  |  ".join(titles))
    model.train()


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class CIFAR100CNN(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 100),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR100(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
val_loader   = DataLoader(test_dataset,  batch_size=64)

# ─────────────────────────────────────────────
# Training — base model
# ─────────────────────────────────────────────
logger = TensorBoardLogger(save_dir="tb_logs", name="cifar100")

model   = CIFAR100CNN()
trainer = L.Trainer(max_epochs=100, logger=logger)

print("\n===== Training base model =====\n")
trainer.fit(model, train_loader, val_loader)

# ── Visualise predictions AFTER base training ──
device = next(model.parameters()).device
raw_writer = logger.experiment          # underlying SummaryWriter
log_prediction_grid(
    raw_writer, model, test_dataset, device,
    tag="BaseModel/Predictions"
)
print("✓ Logged base-model prediction grid to TensorBoard")

# ─────────────────────────────────────────────
# Pruning
# ─────────────────────────────────────────────
print("\n===== Applying pruning =====\n")

PRUNE_AMOUNT = 0.3

for module in [model.features[0], model.features[3],
               model.classifier[1], model.classifier[4]]:
    prune.l1_unstructured(module, name="weight", amount=PRUNE_AMOUNT)

for name, module in model.named_modules():
    if hasattr(module, "weight"):
        try:
            zeros    = torch.sum(module.weight == 0).item()
            total    = module.weight.nelement()
            sparsity = 100 * zeros / total
            print(f"  {name}: {sparsity:.2f}% sparsity")
        except Exception:
            pass

# ─────────────────────────────────────────────
# Fine-tuning after pruning
# ─────────────────────────────────────────────
print("\n===== Fine-tuning =====\n")

fine_tune_logger = TensorBoardLogger(save_dir="tb_logs", name="cifar100_finetuned")
ft_trainer = L.Trainer(max_epochs=5, logger=fine_tune_logger)
ft_trainer.fit(model, train_loader, test_loader)

# ── Visualise predictions AFTER fine-tuning ──
ft_writer = fine_tune_logger.experiment
log_prediction_grid(
    ft_writer, model, test_dataset, device,
    tag="PrunedModel/Predictions"
)
print("✓ Logged pruned-model prediction grid to TensorBoard")

# ─────────────────────────────────────────────
# Remove pruning masks & save
# ─────────────────────────────────────────────
for module in [model.features[0], model.features[3],
               model.classifier[1], model.classifier[4]]:
    prune.remove(module, "weight")

torch.save(model.state_dict(), "cifar100_pruned_finetuned.pt")
print("\n✓ Model saved to cifar100_pruned_finetuned.pt")

# ─────────────────────────────────────────────
# Print TensorBoard log paths
# ─────────────────────────────────────────────
import os
print("\nTensorBoard log directories:")
for root, dirs, files in os.walk("tb_logs"):
    if files:
        print(f"  {root}  ({len(files)} files)")

print(
    "\nTo view results run:\n"
    "  tensorboard --logdir tb_logs\n"
    "Then open http://localhost:6006 in your browser.\n"
    "Check the 'Images' tab for prediction grids,\n"
    "and the 'Scalars' tab for val_acc / val_loss curves."
)