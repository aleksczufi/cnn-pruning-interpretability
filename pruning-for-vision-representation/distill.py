import argparse
import os
import time

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torchvision.models import get_model, get_weight

import utils
from train import create_lr_scheduler, create_optimizer, evaluate, load_data, set_seed
import sys
sys.path.insert(0, "/home/ucu/Dev/igor/pruning")
from explainations_evaluation_metrics import gradCAM, guided_gradCAM, evaluate_single
from glob import glob
from PIL import Image
import numpy as np





def kd_logit_loss(student_logits, teacher_logits, temperature):
    s_log_softmax = F.log_softmax(student_logits / temperature, dim=1)
    t_softmax = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(s_log_softmax, t_softmax, reduction="batchmean") * (temperature ** 2)

def measure_interpretability(model, device, voc_path="/mnt/data/VOC2012/JPEGImages/", num_images=30):
    """Szybki, lekki pomiar interpretowalności po każdej epoce -- mała próbka (30 obrazków),
    żeby nie spowalniać znacząco treningu."""
    model.eval()
    gc_masses, gc_ranks, ggc_masses, ggc_ranks = [], [], [], []

    count = 0
    for img_path in glob(os.path.join(voc_path, "*.jpg")):
        mask_path = img_path.replace("JPEGImages", "SegmentationClass").replace(".jpg", ".png")
        if not os.path.exists(mask_path):
            continue

        from explainations_evaluation_metrics import ImageDataset
        dataset = ImageDataset(img_path)
        for img_tensor, _ in dataset.dataloader:
            image = img_tensor.unsqueeze(0).to(device)
            break

        gt_mask = np.array(Image.open(mask_path))

        class FakeArgs:
            model = "resnet18"
            pruning_iteration = 0
            weights = None
        fake_args = FakeArgs()

        heatmap_gc, _ = gradCAM(model, [model.layer4[-1]], image, fake_args)
        heatmap_ggc = guided_gradCAM(model, model.layer4, image)

        for name, hm, mass_list, rank_list in [
            ("gc", heatmap_gc, gc_masses, gc_ranks),
            ("ggc", heatmap_ggc, ggc_masses, ggc_ranks),
        ]:
            hm_np = hm.squeeze().cpu().detach().numpy() if torch.is_tensor(hm) else hm
            metrics, _ = evaluate_single(heatmap=hm_np, ground_truth=gt_mask.copy(), pooling_type="l2-norm,sq")
            mass_list.append(metrics["mass"])
            rank_list.append(metrics["rank"])

        del image, heatmap_gc, heatmap_ggc
        torch.cuda.empty_cache()

        count += 1
        if count >= num_images:
            break

    model.train()
    return {
        "interp/gc_mass": np.mean(gc_masses),
        "interp/gc_rank": np.mean(gc_ranks),
        "interp/ggc_mass": np.mean(ggc_masses),
        "interp/ggc_rank": np.mean(ggc_ranks),
    }

def train_one_epoch_kd(student, teacher, ce_criterion, optimizer, data_loader,
                        device, epoch, args, scaler=None, global_wandb_step=0):
    student.train()
    teacher.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"KD Epoch: [{epoch}]"

    running_loss, running_ce, running_kd = 0.0, 0.0, 0.0
    tot_acc1, tot_acc5 = 0.0, 0.0
    num_batches = 0

    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.no_grad():
            teacher_logits = teacher(image)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            student_logits = student(image)
            loss_ce = ce_criterion(student_logits, target)
            loss_kd = kd_logit_loss(student_logits, teacher_logits, args.kd_temperature)
            # alpha weights the SOFT/KD term 
            loss = (1.0 - args.kd_alpha) * loss_ce + args.kd_alpha * loss_kd

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = utils.accuracy(student_logits, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_kd += loss_kd.item()
        tot_acc1 += acc1.item()
        tot_acc5 += acc5.item()
        num_batches += 1

    if utils.is_main_process():
        wandb.log({"train/acc1": tot_acc1 / num_batches}, step=global_wandb_step)
        wandb.log({"train/acc5": tot_acc5 / num_batches}, step=global_wandb_step)
        wandb.log({"train/loss": running_loss / num_batches}, step=global_wandb_step)
        wandb.log({"train/ce_loss": running_ce / num_batches}, step=global_wandb_step)
        wandb.log({"train/kd_loss": running_kd / num_batches}, step=global_wandb_step)


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)
    args.distributed = False

    if args.seed is not None:
        set_seed(args.seed)

    if utils.is_main_process():
        wandb.init(
            project=f"kd-{args.student}-from-{args.teacher}",
            name=f"kd-{args.teacher}-to-{args.student}-T{args.kd_temperature}-a{args.kd_alpha}",
            config={
                "teacher": args.teacher,
                "student": args.student,
                "epochs": args.epochs,
                "kd_temperature": args.kd_temperature,
                "kd_alpha": args.kd_alpha,
                "kd_alpha_convention": "alpha weights KD (soft-label) term, T^2-scaled, Hinton et al. convention",
            },
        )

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"Loading teacher: {args.teacher} (pretrained, frozen)")
    teacher_weights = get_weight(args.teacher_weights) if args.teacher_weights else "DEFAULT"
    teacher = get_model(args.teacher, weights=teacher_weights).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Creating student: {args.student} ({'pretrained' if args.student_pretrained else 'random init'})")
    student = get_model(args.student, weights="DEFAULT" if args.student_pretrained else None).to(device)


    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = create_optimizer(args=args, parameters=student.parameters())
    lr_scheduler = create_lr_scheduler(args=args, optimizer=optimizer)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None


    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        _original_torch_load = torch.load
        def _torch_load_weights_only_false(*a, **kw):
            kw["weights_only"] = False
            return _original_torch_load(*a, **kw)
        torch.load = _torch_load_weights_only_false
        try:
            ckpt = torch.load(args.resume, map_location=device)
        finally:
            torch.load = _original_torch_load

        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        args.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {args.start_epoch}")


    print("Evaluating teacher baseline (sanity check against torchvision's reported accuracy)...")
    evaluate(teacher, ce_criterion, data_loader_test, device=device, log_suffix="[teacher]")

    print("Evaluating student baseline (pre-distillation)...")
    evaluate(student, ce_criterion, data_loader_test, device=device, log_suffix="[student-pre]")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_kd(
            student=student, teacher=teacher, ce_criterion=ce_criterion,
            optimizer=optimizer, data_loader=data_loader, device=device,
            epoch=epoch, args=args, scaler=scaler, global_wandb_step=epoch,
        )
        lr_scheduler.step()
        evaluate(student, ce_criterion, data_loader_test, device=device, global_wandb_step=epoch)

        interp_metrics = measure_interpretability(student, device)
        if utils.is_main_process():
            wandb.log(interp_metrics, step=epoch)
        
        print(f"Epoch {epoch} interpretability: {interp_metrics}")
        if args.output_dir:
            checkpoint = {
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()

            if epoch == args.epochs - 1 or (epoch % 10 == 0):
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, f"{args.student}_kd_from_{args.teacher}_epoch_{epoch}.pth"),
                )
            utils.save_on_master(
                checkpoint,
                os.path.join(
                    args.output_dir,
                    f"{args.student}_kd_from_{args.teacher}_T{args.kd_temperature}_a{args.kd_alpha}_checkpoint.pth",
                ),
            )

    total_time = time.time() - start_time
    print(f"KD training time: {total_time / 3600:.2f} hours")


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/mnt/data/ImageNet/ILSVRC/Data/CLS-LOC", type=str)
    parser.add_argument("--teacher", default="resnet152", type=str)
    parser.add_argument("--teacher-weights", default=None, type=str, help="e.g. ResNet152_Weights.IMAGENET1K_V2")
    parser.add_argument("--student", default="resnet18", type=str)
    parser.add_argument("--student-pretrained", action="store_true",
                         help="init student from ImageNet weights instead of random")

    # KD hyperparameters 
    parser.add_argument("--kd-temperature", default=4.0, type=float)
    parser.add_argument("--kd-alpha", default=0.5, type=float,
                         help="weight on soft-label KD (KL-divergence) term, Hinton convention; "
                              "(1 - alpha) is applied to the hard-label CE term")

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("-j", "--workers", default=16, type=int)
    parser.add_argument("--opt", default="sgd", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight-decay", default=5e-4, type=float, dest="weight_decay")
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str)
    parser.add_argument("--lr-warmup-epochs", default=0, type=int)
    parser.add_argument("--lr-warmup-method", default="linear", type=str)
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float)
    parser.add_argument("--lr-step-size", default=30, type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--lr-min", default=0.0, type=float)
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--output-dir", default="./results_kd", type=str)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--resume", default="", type=str, help="Path to a checkpoint to resume training from (loads model, optimizer, "
                          "lr_scheduler state, and sets start_epoch automatically).")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--interpolation", default="bilinear", type=str)
    parser.add_argument("--val-resize-size", default=256, type=int)
    parser.add_argument("--val-crop-size", default=224, type=int)
    parser.add_argument("--train-crop-size", default=224, type=int)
    parser.add_argument("--backend", default="PIL", type=str.lower)
    parser.add_argument("--use-v2", action="store_true")
    parser.add_argument("--norm-weight-decay", default=None, type=float)
    parser.add_argument("--bias-weight-decay", default=None, type=float)
    parser.add_argument("--transformer-embedding-decay", default=None, type=float)
    parser.add_argument("--mixup-alpha", default=0.0, type=float)
    parser.add_argument("--cutmix-alpha", default=0.0, type=float)
    parser.add_argument("--auto-augment", default=None, type=str)
    parser.add_argument("--ra-magnitude", default=9, type=int)
    parser.add_argument("--augmix-severity", default=3, type=int)
    parser.add_argument("--random-erase", default=0.0, type=float)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--test-only", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)