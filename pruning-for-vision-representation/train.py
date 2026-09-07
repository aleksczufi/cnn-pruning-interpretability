import datetime
import os
import time
import warnings

import torch.nn.utils.prune as prune
import presets
import wandb
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
        
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, split='train', global_wandb_step=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    tot_acc1 = 0
    tot_acc5 = 0
    tot_loss = 0

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        tot_acc1 += acc1.item()
        tot_acc5 += acc5.item()
        tot_loss += loss

    if utils.is_main_process():
        wandb.log({f"{split}/acc1": tot_acc1/len(data_loader)}, step=global_wandb_step)
        wandb.log({f"{split}/acc5": tot_acc5/len(data_loader)}, step=global_wandb_step)
        wandb.log({f"{split}/loss": tot_loss/len(data_loader)}, step=global_wandb_step)


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", split='test', global_wandb_step=0):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    tot_acc1 = 0
    tot_acc5 = 0
    tot_loss = 0   

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

            tot_acc1 += acc1.item()
            tot_acc5 += acc5.item()
            tot_loss += loss

    if utils.is_main_process():
        wandb.log({f"{split}/acc1": tot_acc1/len(data_loader)}, step=global_wandb_step)
        wandb.log({f"{split}/acc5": tot_acc5/len(data_loader)}, step=global_wandb_step)
        wandb.log({f"{split}/loss": tot_loss/len(data_loader)}, step=global_wandb_step)

    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def snip_pruning(model, data_loader, device, criterion, target_sparsity=0.9):
    """
    Single-shot Network Pruning based on Connection Sensitivity
    """
    print(f"Applying SNIP pruning with target sparsity {target_sparsity}...")
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, targets = next(data_iter)
    images = images.to(device)
    targets = targets.to(device)
    
    # Register hooks and collect all prunable parameters
    parameters_to_prune = []
    handles = []
    grads = {}
    
    def hook_factory(name):
        def hook(grad):
            grads[name] = grad.detach().clone().abs()
        return hook
    
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
            if hasattr(module, 'weight'):
                param = module.weight
                handle = param.register_hook(hook_factory(f"{id(module)}.weight"))
                handles.append(handle)
    
    # Forward and backward pass to get gradient information
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Remove the hooks
    for handle in handles:
        handle.remove()
    
    # Compute saliency scores
    scores = {}
    all_scores = []
    for module, param_name in parameters_to_prune:
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            grad_key = f"{id(module)}.{param_name}"
            if grad_key in grads:
                score = param.abs() * grads[grad_key]
                scores[grad_key] = score
                all_scores.append(score.view(-1))
    
    # Flatten and concatenate all scores
    all_scores_tensor = torch.cat(all_scores)
    
    # Compute threshold for target sparsity
    # Note: We want to keep the TOP (1-target_sparsity) connections, so we find
    # the threshold that prunes target_sparsity connections
    k = int(all_scores_tensor.numel() * target_sparsity)
    if k >= all_scores_tensor.numel():
        threshold = float('inf')  # Prune everything if k is too large
    elif k <= 0:
        threshold = -1  # Keep everything if k is 0 or negative
    else:
        # Sort scores and find the k-th smallest value
        sorted_scores, _ = torch.sort(all_scores_tensor)
        threshold = sorted_scores[k-1].item()
    
    print(f"SNIP threshold: {threshold}")
    
    # Apply masks based on scores (prune connections with scores <= threshold)
    for module, param_name in parameters_to_prune:
        module_param_key = f"{id(module)}.{param_name}"
        if module_param_key in scores:
            score = scores[module_param_key]
            mask = (score > threshold).float()
            prune.custom_from_mask(module, param_name, mask)
    
    return model


def magnitude_pruning(model, prune_amount=0.2):
    """
    Magnitude-based pruning
    
    Args:
        model: The neural network model
        prune_amount: Amount of connections to prune (between 0 and 1)
    
    Returns:
        Pruned model
    """
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount
    )
    
    return model


def compute_sparsity_global(model):
    """
    Compute the global sparsity of the model
    
    Args:
        model: The neural network model
    
    Returns:
        float: Sparsity percentage (0-100)
    """
    total_params = 0
    zero_params = 0
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            total_params += module.weight.nelement()
            zero_params += torch.sum(module.weight == 0).item()
    
    if total_params == 0:
        return 0.0
        
    sparsity = 100.0 * zero_params / total_params
    return sparsity


def create_optimizer(args, parameters):
    print('Initializing Optimizer ...')
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    
    return optimizer


def create_lr_scheduler(args, optimizer):
    print('Initializing Lr Scheduler...')
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler


def train_model_to_completion(model, data_loader, data_loader_test, criterion, args, device, scaler=None, initial_epoch=0, model_ema=None, global_wandb_step_offset=0):
    """
    Standard training function to train a model to completion
    """
    print("Starting standard training to completion")
    
    # Get model without DDP wrapper if needed
    model_without_ddp = model
    if args.distributed:
        model_without_ddp = model.module
    
    # Set up optimizer and learning rate scheduler
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
            
    parameters = utils.set_weight_decay(
        model_without_ddp,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    optimizer = create_optimizer(args=args, parameters=parameters)
    lr_scheduler = create_lr_scheduler(args=args, optimizer=optimizer)
    
    # Track current sparsity
    sparsity = compute_sparsity_global(model_without_ddp)
    print(f"Starting training with sparsity: {sparsity:.2f}%")
    
    # Train the model
    start_time = time.time()
    for epoch in range(initial_epoch, args.epochs):
        global_wandb_step = epoch + global_wandb_step_offset
        
        if utils.is_main_process():
            wandb.log({'Epoch': epoch}, step=global_wandb_step)
            
        if args.distributed:
            if hasattr(args, "train_sampler"):
                args.train_sampler.set_epoch(epoch)
            else:
                print("Warning: train_sampler not found in args")
                
        train_one_epoch(
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            data_loader=data_loader, 
            device=device, 
            epoch=epoch, 
            args=args, 
            model_ema=model_ema, 
            scaler=scaler, 
            global_wandb_step=global_wandb_step
        )
        
        if utils.is_main_process():
            wandb.log({'Learning rate': optimizer.param_groups[0]["lr"]}, step=global_wandb_step)
            wandb.log({'sparsity': sparsity}, step=global_wandb_step)
        
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device, global_wandb_step=global_wandb_step)
        
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", global_wandb_step=global_wandb_step)
        
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "sparsity": sparsity,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
                
            if epoch == args.epochs - 1 or (epoch % 10 == 0):
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"{args.model}_epoch_{epoch}_{args.pruning_method}_{args.target_sparsity}.pth"))
                
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"{args.model}_checkpoint_{args.pruning_method}_{args.target_sparsity}.pth"))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    
    return model, sparsity


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    print(device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if utils.is_main_process():
        run = wandb.init(
            project=f"snip-{args.model}-cassano_tesi",
            name=f"{args.model}-pruning-{args.pruning_method}",
            config={
                "architecture": str(args.model),
                "dataset": "Imagenet-1K",
                "epochs": args.epochs,
                "pruning_method": args.pruning_method,
                "target_sparsity": args.target_sparsity
            }
        )

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    
    # Store train_sampler in args for later use
    args.train_sampler = train_sampler
    
    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    if args.seed is not None:
        set_seed(args.seed)

    print("Creating model")
    if 'vit' in args.model:
        model = torchvision.models.vit_b_32(weights=None, num_classes=num_classes)
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Get model without DDP wrapper if needed
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    if args.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, data_loader_test, device=device)
        return
    
    # Different pruning strategies
    if args.pruning_method == "snip":
        # ===== SNIP: Prune once at the beginning, then train to completion =====
        
        # Apply SNIP pruning
        model_without_ddp = snip_pruning(
            model=model_without_ddp,
            data_loader=data_loader,
            device=device,
            criterion=criterion,
            target_sparsity=args.target_sparsity
        )
        
        # Get current sparsity
        sparsity = compute_sparsity_global(model_without_ddp)
        print(f"Sparsity after SNIP pruning: {sparsity:.2f}%")
        
        # Create EMA model if needed
        model_ema = None
        if args.model_ema:
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
            
        # Train the pruned model to completion
        _, final_sparsity = train_model_to_completion(
            model=model,
            data_loader=data_loader,
            data_loader_test=data_loader_test,
            criterion=criterion,
            args=args,
            device=device,
            scaler=scaler,
            model_ema=model_ema
        )
        
        print(f"Final sparsity after SNIP and training: {final_sparsity:.2f}%")
        
    elif args.pruning_method == "magnitude":
        # ===== LRR: Iterative pruning and training =====
        pruning_thresh = args.pruning_threshold  # Target sparsity threshold
        prune_iter_count = args.starting_pruning_iteration
        
        # Initialize sparsity
        sparsity = compute_sparsity_global(model_without_ddp)
        print(f"Initial sparsity: {sparsity:.2f}%")
        
        # Iterative pruning and training
        while sparsity < pruning_thresh:
            print(f'Pruning iteration: {prune_iter_count}')
            
            # Create EMA model if needed
            model_ema = None
            if args.model_ema:
                adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
                alpha = 1.0 - args.model_ema_decay
                alpha = min(1.0, alpha * adjust)
                model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
            
            # Train model for this pruning iteration
            global_wandb_step_offset = args.epochs * prune_iter_count
            _, sparsity = train_model_to_completion(
                model=model,
                data_loader=data_loader,
                data_loader_test=data_loader_test,
                criterion=criterion,
                args=args,
                device=device,
                scaler=scaler,
                model_ema=model_ema,
                global_wandb_step_offset=global_wandb_step_offset
            )
            
            # Apply magnitude pruning
            n_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            
            model_without_ddp = magnitude_pruning(
                model=model_without_ddp,
                prune_amount=args.pruning_rate
            )
            
            # Update sparsity
            sparsity = compute_sparsity_global(model_without_ddp)
            print(f'Current Sparsity: {sparsity:.2f}%')
            print(f'Target Pruning Threshold: {pruning_thresh}%')
            
            # Increment pruning iteration counter
            prune_iter_count += 1
            
            torch.set_num_threads(n_threads)
    
    else:
        raise ValueError(f"Unsupported pruning method: {args.pruning_method}. Choose 'snip' or 'magnitude'.")
    
    print("Training completed successfully")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training with Pruning", add_help=add_help)

    parser.add_argument("--data-path", default="/shared/datasets/classification/imagenet/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    
    # Pruning configuration
    parser.add_argument(
        "--pruning-method",
        default="magnitude",
        type=str,
        choices=["magnitude", "snip"],
        help="pruning method to use (magnitude or snip)",
    )
    parser.add_argument(
        "--target-sparsity",
        default=0.9,
        type=float,
        help="target sparsity for SNIP pruning (default: 0.9)",
    )
    parser.add_argument(
        "--pruning-rate",
        default=0.2,
        type=float,
        help="pruning rate per iteration for magnitude pruning (default: 0.2)",
    )
    parser.add_argument(
        "--pruning-threshold",
        default=95.0,
        type=float,
        help="target pruning threshold for magnitude pruning (default: 95.0)",
    )
    parser.add_argument(
        "--starting-pruning-iteration",
        default=0,
        type=int,
        help="starting pruning iteration for magnitude pruning (default: 0)",
    )
    
    # Standard training arguments
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    