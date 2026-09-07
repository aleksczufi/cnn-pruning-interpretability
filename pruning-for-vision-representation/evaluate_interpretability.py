"""
Usage (student, from a distill.py checkpoint):
    python evaluate_interpretability.py \
        --checkpoint-path ./results_kd/resnet18_kd_from_resnet152_T4.0_a0.5_checkpoint.pth \
        --model resnet18 \
        --voc-path /mnt/data/VOC2012/JPEGImages/ \
        --output-dir ./interp_results \
        --num-images 101 \
        --label student_kd_T4_a0.5

Usage (teacher, pretrained torchvision weights, no checkpoint file):
    python evaluate_interpretability.py \
        --pretrained \
        --model resnet152 \
        --voc-path /mnt/data/VOC2012/JPEGImages/ \
        --output-dir ./interp_results \
        --num-images 101 \
        --label teacher_resnet152
"""

import argparse
import os
from glob import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from explainations_evaluation_metrics import (
    ImageDataset,
    evaluate_single,
    gradCAM,
    guided_gradCAM,
    integrated_gradients,
    load_model,
)


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Interpretability eval on a direct checkpoint path or pretrained teacher", add_help=add_help)
    parser.add_argument("--checkpoint-path", default=None, type=str,
                         help="Direct path to a .pth checkpoint (e.g. from distill.py). "
                              "Required unless --pretrained is set.")
    parser.add_argument("--pretrained", action="store_true",
                         help="Skip checkpoint loading entirely and use torchvision's pretrained "
                              "ImageNet weights directly -- use this for evaluating a frozen teacher "
                              "(e.g. resnet50, resnet152) with no distillation checkpoint involved.")
    parser.add_argument("--model", default="resnet18", type=str,
                         help="Architecture name, must match what gradCAM/load_model expect "
                              "(e.g. 'resnet18'; determines which layer gradCAM hooks into)")
    parser.add_argument("--voc-path", default="/mnt/data/VOC2012/JPEGImages/", type=str,
                         help="Path to VOC2012 JPEGImages/ dir (masks are found by string-replacing "
                              "'JPEGImages'->'SegmentationClass' and '.jpg'->'.png')")
    parser.add_argument("--output-dir", default="./interp_results", type=str)
    parser.add_argument("--num-images", default=101, type=int,
                         help="Max number of VOC images (with masks) to evaluate on")
    parser.add_argument("--pooling-type", default="l2-norm,sq", type=str,
                         help="Channel-pooling method for heatmaps before evaluate_single")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--label", default="", type=str,
                         help="Free-text tag for this run (e.g. 'student_kd_T4_a0.5' or 'teacher_resnet152'), "
                              "used in output filenames/CSV rows")
    parser.add_argument("--weights", default=None, type=str,
                         help="torchvision weights enum name for get_model(), e.g. 'ResNet18_Weights.IMAGENET1K_V1'. "
                              "In checkpoint mode this only affects the initial (soon-to-be-overwritten) model "
                              "construction inside load_model(). Ignored in --pretrained mode (uses 'DEFAULT' instead).")
    parser.add_argument("--pruning-iteration", default=0, type=int,
                         help="Unused by this script's own logic, but referenced internally by "
                              "load_model()/gradCAM() -- kept here only for argument compatibility.")
    parser.add_argument("--skip-ig", action="store_true",
                         help="Skip Integrated Gradients entirely. IG with noise-tunnel smoothing "
                              "runs many forward/backward passes per image internally and can be "
                              "significantly more memory-hungry than Grad-CAM/Guided Grad-CAM, "
                              "especially on deeper models (e.g. resnet50/152) or when sharing the "
                              "GPU with another process. Use this flag if you hit CUDA OOM errors "
                              "specifically inside integrated_gradients().")
    return parser


def load_pretrained_teacher(args, device):
    """
    Loads a torchvision model with its default pretrained ImageNet weights.
    There's no checkpoint file to load for an off-the-shelf teacher, just
    torchvision's built-in weights.
    """
    print(f"Loading pretrained torchvision weights for '{args.model}' (no checkpoint file)")
    model = torchvision.models.get_model(args.model, weights="DEFAULT")
    return model


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.pretrained:
        # Teacher mode: no checkpoint file.
        model = load_pretrained_teacher(args, device)
    else:
        if not args.checkpoint_path:
            raise ValueError("Must pass either --checkpoint-path or --pretrained")
        _original_torch_load = torch.load

        def _torch_load_weights_only_false(*load_args, **load_kwargs):
            load_kwargs["weights_only"] = False
            return _original_torch_load(*load_args, **load_kwargs)

        torch.load = _torch_load_weights_only_false
        try:
            print(f"Loading model '{args.model}' from checkpoint: {args.checkpoint_path}")
            model = load_model(args.checkpoint_path, args, device=device, num_classes=1000)
        finally:
            torch.load = _original_torch_load

    model.eval()
    model = model.to(device)

    # Gather VOC images that have a corresponding segmentation mask 
    # Only images with an existing mask are kept.
    imgs, masks, image_paths, mask_paths = [], [], [], []
    skipped = 0

    for single_image_path in glob(os.path.join(args.voc_path, "*.jpg")):
        mask_path = single_image_path.replace("JPEGImages", "SegmentationClass").replace(".jpg", ".png")
        if not os.path.exists(mask_path):
            skipped += 1
            continue

        dataset = ImageDataset(single_image_path)
        for img, path in dataset.dataloader:
            imgs.append(img.unsqueeze(0))
            image_paths.append(path)
            mask_paths.append(mask_path)
            break

        # Load the mask as a raw class-index array, not through ImageDataset --
        # ImageDataset applies photo-style preprocessing (resize/normalize/ToTensor)
        # meant for natural images, which corrupts VOC's palette-mode segmentation
        # PNGs (each pixel value = a class index, not RGB color data).
        mask_img = Image.open(mask_path)
        mask_np = np.array(mask_img)
        masks.append(mask_np)

        if len(imgs) >= args.num_images:
            break

    print(f"Collected {len(imgs)} images with masks (skipped {skipped} without masks)")

    # Run attribution methods + evaluate_single per image 
    results = {
        "gc_mass": [], "gc_rank": [],
        "ggc_mass": [], "ggc_rank": [],
    }
    if not args.skip_ig:
        results["ig_mass"] = []
        results["ig_rank"] = []

    for idx, image in enumerate(tqdm(imgs)):
        image = image.to(device)
        gt_mask = masks[idx]

        heatmap_gc, pred = gradCAM(model, [model.layer4[-1]], image, args)
        heatmap_ggc = guided_gradCAM(model, model.layer4, image)

        attribution_methods = [("gc", heatmap_gc), ("ggc", heatmap_ggc)]
        if not args.skip_ig:
            heatmap_ig = integrated_gradients(model, image)
            attribution_methods.append(("ig", heatmap_ig))

        for name, heatmap in attribution_methods:
            heatmap_np = heatmap.squeeze().cpu().detach().numpy() if torch.is_tensor(heatmap) else heatmap
            metrics, _ = evaluate_single(heatmap=heatmap_np, ground_truth=gt_mask.copy(), pooling_type=args.pooling_type)
            results[f"{name}_mass"].append(metrics["mass"])
            results[f"{name}_rank"].append(metrics["rank"])

        # Free GPU memory accumulated by this image's attribution computations
        # (Integrated Gradients in particular can accumulate significant memory
        # across noise-tunnel samples/interpolation steps) before moving on to
        # the next image --without this, memory usage can climb across the
        # loop and eventually exhaust the GPU, especially when sharing the GPU
        # with another running process (e.g. a concurrent training job).
        del image, heatmap_gc, heatmap_ggc
        if not args.skip_ig:
            del heatmap_ig
        torch.cuda.empty_cache()

    # Save per-image results + summary
    import csv
    out_csv = os.path.join(args.output_dir, f"interpretability_{args.label or args.model}.csv")
    keys = list(results.keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"] + keys)
        for i, path in enumerate(image_paths):
            writer.writerow([path] + [results[k][i] for k in keys])

    print(f"\nSaved per-image results to: {out_csv}")
    print("\nMean metrics across all evaluated images:")
    for k in keys:
        vals = results[k]
        print(f"  {k}: {np.mean(vals):.4f} (n={len(vals)})")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)