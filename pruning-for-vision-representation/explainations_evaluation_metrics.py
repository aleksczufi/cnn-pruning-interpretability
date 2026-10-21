"""
Code to evaluate explanation methods (aka heatmaps/XAI methods) w.r.t. ground truths, as in:
Towards Ground Truth Evaluation of Visual Explanations, Osman et al. 2020
https://arxiv.org/pdf/2003.07258.pdf
(code by Leila Arras, Fraunhofer HHI - Berlin, Germany)
"""
from typing import Dict, List
import os, json, sys, math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from matplotlib.colors import Normalize

import torchvision
from glob import glob
import torch.nn.utils.prune as prune
from tqdm import tqdm
from sklearn.preprocessing import normalize

import vision_transformer_no_save as vision_transformer
from vision_transformer_no_save import interpolate_embeddings

from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms as pth_transforms
import presets

import torchvision.transforms
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import LayerAttribution
from captum.attr import LRP
from captum.attr import LayerGradCam 
from captum.attr import NoiseTunnel
from captum.attr import GuidedGradCam 
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from imagenet_classes import IMAGENET2012_CLASSES

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from collections import OrderedDict

from datasets import ImageDataset, Dataset, bbox_iou

def interpolate_vit(base_vit, device, img_shape, args, first=False):

    model = base_vit
    # Rendo il pruning permanente, necessario per le key
    if args.pruning_iteration > 0 and not first:
        pruned_layer_types =  [torch.nn.Conv2d, torch.nn.Linear]
        for module in model.modules():
            # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
            for t in pruned_layer_types:
                if isinstance(module, t) and prune.is_pruned(module): 
                    prune.remove(module, "weight")
                module.to(device)


    print('Shape passata')
    print(img_shape)

    patch_size = 32
    # model_interpolated = torchvision.models.vit_b_32(pretrained=False, image_size=img_shape)
    model_interpolated = vision_transformer.vit_b_32(pretrained=False, image_size=img_shape)


    state_dict_new = interpolate_embeddings(img_shape, patch_size, model.state_dict())
    model_interpolated.load_state_dict(state_dict_new)

    if args.pruning_iteration > 0 and not first:
        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        for module in model_interpolated.modules():
            # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
            for t in pruned_layer_types:
                if isinstance(module, t): 
                    prune.identity(module, "weight")
                module.to(device)

    model = model_interpolated

    model.eval()
    model.to(device)

    return model

def load_model(checkpoint_path, args, device, num_classes=1000, base_vit_interpolated = None):

    if args.model == 'swin':
        model = torchvision.models.swin_v2_t()
        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
    else:
        if 'vit' in args.model:
            if base_vit_interpolated is not None:
                model = base_vit_interpolated
            else:
                # model = torchvision.models.vit_b_32(image_size=224)
                model = vision_transformer.vit_b_32(image_size=224)
        else:
            model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)

        if args.model == 'resnet50':
            pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        elif 'vit' in args.model:
            pruned_layer_types = [torch.nn.Linear, torch.nn.Conv2d] # , torch.nn.Linear.NonDynamicallyQuantizableLinear]
        else:
            pruned_layer_types = [torch.nn.Conv2d]

    print('Pruned Layers Type')
    print(pruned_layer_types)
    for child in model.named_children():
        print(child)


    checkpoint = torch.load(checkpoint_path, map_location=device)["model"]

    correct_checkpoint = OrderedDict()
    if 'vit' in args.model:
        for k in checkpoint:
            correct_checkpoint[k.replace("module.", "")] = checkpoint[k]

        try:
            model.load_state_dict(correct_checkpoint)
        except:
            # for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
            for module in model.modules():
                # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
                for t in pruned_layer_types:
                    if isinstance(module, t): 
                        prune.identity(module, "weight")
                    module.to(device)
            
    
            try:
                model.load_state_dict(correct_checkpoint)
    
#                 for module in model.modules():
#                     # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
#                     for t in pruned_layer_types:
#                         if isinstance(module, t): 
#                             prune.remove(module, "weight")
#                         module.to(device)
                for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
                    prune.remove(module, "weight")
                    module.to(device)
            except:
                raise RuntimeError()
    else:
        for k in checkpoint:
            correct_checkpoint[k.replace("module.", "")] = checkpoint[k]
            # print(correct_checkpoint)
            # exit()
    
        try:
            model.load_state_dict(correct_checkpoint)
        except:
            for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
                prune.identity(module, "weight")
            
    
            try:
                model.load_state_dict(correct_checkpoint)
    
                for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
                    prune.remove(module, "weight")
            except:
                raise RuntimeError()

    # Per cuda out of memory
#    for param in model.features.parameters():
#        param.requires_grad = False

#     new_dict = interpolate_embeddings(image_size=(512, 512), patch_size=32, model_state=model.state_dict())
#     model = torchvision.models.vit_b_32(image_size=512)
#     model.load_state_dict(new_dict)

    model.eval()
    model.to(device)

    return model

def pool_heatmap (heatmap: np.ndarray, pooling_type: str) -> np.ndarray:
    """
    Pool the relevance along the channel axis, according to the pooling technique specified by pooling_type.
    """
    C, H, W = heatmap.shape

    if pooling_type=="sum,abs":
        pooled_heatmap = np.abs(np.sum(heatmap, axis=0))

    elif pooling_type=="sum,pos":
        pooled_heatmap = np.sum(heatmap, axis=0) ; pooled_heatmap = np.where(pooled_heatmap>0.0, pooled_heatmap, 0.0)
    
    elif pooling_type=="max-norm":
        pooled_heatmap = np.amax(np.abs(heatmap), axis=0)

    elif pooling_type=="l1-norm":
        pooled_heatmap = np.linalg.norm(heatmap, ord=1, axis=0)

    elif pooling_type=="l2-norm":
        pooled_heatmap = np.linalg.norm(heatmap, ord=2, axis=0)

    elif pooling_type=="l2-norm,sq":
        pooled_heatmap = (np.linalg.norm(heatmap, ord=2, axis=0))**2

    assert pooled_heatmap.shape == (H, W) and np.all(pooled_heatmap>=0.0)
    return pooled_heatmap


def evaluate_single (heatmap: np.ndarray, ground_truth: np.ndarray, pooling_type: str) -> Dict[str, np.float64]:
    """
    Given an image's relevance heatmap and a corresponding ground truth boolean ndarray of the same vertical and horizontal dimensions, return both:
     - the ratio of relevance falling into the ground truth area w.r.t. the total amount of relevance ("relevance mass accuracy" metric)
     - the ratio of pixels within the N highest relevant pixels (where N is the size of the ground truth area) that effectively belong to the ground truth area
       ("relevance rank accuracy" metric)
    Both ratios are calculated after having pooled the relevance across the channel axis, according to the pooling technique defined by the pooling_type argument.
    Args:
    - heatmap (np.ndarray):         of shape (C, H, W), with dtype float 
    - ground_truth (np.ndarray):    of shape (H, W), with dtype bool
    - pooling_type (str):           specifies how to pool the relevance across the channels, i.e. defines a mapping function f: R^C -> R^+
                                    that maps a real-valued vector of dimension C to a positive number (see details of each pooling_type in the function pool_heatmap)
    Returns:
    A dict wich keys=["mass", "rank"] and resp. values:
    - relevance_mass_accuracy (np.float64):     relevance mass accuracy, float in the range [0.0, 1.0], the higher the better.
    - relevance_rank_accuracy (np.float64):     relevance rank accuracy, float in the range [0.0, 1.0], the higher the better.
    """

    print('Shape')
    print(ground_truth.shape)
    print(heatmap.shape)

    if len(heatmap.shape) > 2:
        heatmap = heatmap[:, :ground_truth.shape[0], :ground_truth.shape[1]]
        C, H, W = heatmap.shape # C relevance values per pixel coordinate (C=number of channels), for an image with vertical and horizontal dimensions HxW
    else:
        # heatmap = np.transpose(heatmap)
        # heatmap = heatmap.reshape((ground_truth.shape[0], ground_truth.shape[1]))
        print(type(heatmap))
        size_im = (
            heatmap.shape[0],
            heatmap.shape[1],
        )
        paded = torch.zeros(size_im)
        paded = paded.detach().cpu().numpy()
        paded[: ground_truth.shape[0], : ground_truth.shape[1]] = ground_truth
        ground_truth = paded

        H, W = heatmap.shape # C relevance values per pixel coordinate (C=number of channels), for an image with vertical and horizontal dimensions HxW

    print('Shape')
    print(ground_truth.shape)
    print(heatmap.shape)
    print(type(heatmap))
    assert ground_truth.shape == (H, W)

    heatmap = heatmap.astype(dtype=np.float64) # cast heatmap to float64 precision (better for computing relevance accuracy statistics)
    
    # step 1: pool the relevance across the channel dimension to get one positive relevance value per pixel coordinate
    if pooling_type is not None:
        pooled_heatmap = pool_heatmap (heatmap, pooling_type)
    else:
        normalized_mh = (heatmap-np.min(heatmap))/((np.max(heatmap)-np.min(heatmap))+sys.float_info.epsilon)
        pooled_heatmap = normalized_mh # pool_heatmap (heatmap, pooling_type)

    # Aggiunta mia perchè le ground truth usate non rispettano il fatto di essere tutti valori positivi.
    # pooled_ground_truth = pool_heatmap(ground_truth.reshape((1, ground_truth.shape[0], ground_truth.shape[1])), pooling_type)
    # ground_truth = pooled_ground_truth

    # Alternativa: clippo i valori fra zero ed 1
    clipped_ground_truth =  np.clip(ground_truth, a_min = 0, a_max = 1) 
    # ground_truth = clipped_ground_truth

    # Alternativa: normalizzo i valori.
    # normalized_ground_truth = normalize(ground_truth, axis=0)
    normalized_ground_truth = (ground_truth-np.min(ground_truth))/(np.max(ground_truth)-np.min(ground_truth))
    normalized_ground_truth[normalized_ground_truth >= .5] = 1
    normalized_ground_truth[normalized_ground_truth < .5] = 0
    ground_truth = normalized_ground_truth 

    # step 2: compute the ratio of relevance mass within ground truth w.r.t the total relevance
    relevance_within_ground_truth = np.sum(pooled_heatmap * np.where(ground_truth, 1.0, 0.0).astype(dtype=np.float64) )
    print(relevance_within_ground_truth)
    relevance_total               = np.sum(pooled_heatmap) + sys.float_info.epsilon
    print(relevance_total)

    # relevance_mass_accuracy       = 1.0 * relevance_within_ground_truth/relevance_total; print(relevance_mass_accuracy) ; assert (0.0<=relevance_mass_accuracy) and (relevance_mass_accuracy<=1.0)

    relevance_mass_accuracy       = 1.0 * relevance_within_ground_truth/relevance_total; print("RMA: " + str(relevance_mass_accuracy)) ; assert (0.0<=relevance_mass_accuracy) and (relevance_mass_accuracy<=1.1)

    # step 3: order the pixel coordinates in decreasing order of their relevance, then count the number N_gt of pixels within the N highest relevant pixels that fall 
    # into the ground truth area, where N is the total number of pixels of the ground truth area, then compute the ratio N_gt/N
    pixels_sorted_by_relevance = np.argsort(np.ravel(pooled_heatmap))[::-1] ; assert pixels_sorted_by_relevance.shape == (H*W,) # sorted pixel indices over flattened array
    gt_flat = np.ravel(ground_truth)                                        ; assert gt_flat.shape == (H*W,) # flattened ground truth array
    N    = np.sum(gt_flat)  + sys.float_info.epsilon
    print(N)
    N_gt = np.sum(gt_flat[pixels_sorted_by_relevance[:int(N)]])
    print(N_gt)
    # relevance_rank_accuracy = 1.0 * N_gt/N ; print(relevance_rank_accuracy) ; assert (0.0<=relevance_rank_accuracy) and (relevance_rank_accuracy<=1.0)
    relevance_rank_accuracy = 1.0 * N_gt/N ; print(relevance_rank_accuracy) ; assert (0.0<=relevance_rank_accuracy) and (relevance_rank_accuracy<=1.1)
        
    return {"mass": relevance_mass_accuracy, "rank": relevance_rank_accuracy}, ground_truth # dict of relevance accuracies, with key=evaluation metric


def evaluate (heatmap_DIR: str, ground_truth_DIR: str, output_DIR: str, idx_list: List[int], output_name: str = "", evaluation_metric: str = "rank"):
    """
    Given a set of relevance heatmaps obtained via an explanation method (e.g. via LRP or Integrated Gradients), and a set of ground truth masks, 
    both previously saved to disk as numpy ndarrays (resp. in the directories heatmap_DIR and ground_truth_DIR, and with filenames <data point idx>+".npy"), 
    compute various relevance accuracy metric statistics over the subset of data points specified by their indices via the argument idx_list.
    The statistics over this subset of data points, as well as the relevance accuracies for each data point, will be saved to disk in the directory output_DIR as JSON files.
    Additionally, two text files will be written to the directory output_DIR, they contain the summary statistics for different types of pooling ordered by their mean value
    in decreasing order, or printed in a pre-defined fixed order, and formatted as tables.
    The relevance accuracy metric used for evaluating the heatmaps is specified via the evaluation_metric argument, it can be either "mass" or "rank".
    """
  
    accuracy = {} # key=idx of data point (as string), value=dict containing one relevance accuracy per pooling type (with pooling_type as key, and relevance accuracy as value)

    # iterate over data points
    for idx in idx_list:

        heatmap      = np.load(os.path.join(heatmap_DIR,      str(idx) + ".npy"))
        ground_truth = np.load(os.path.join(ground_truth_DIR, str(idx) + ".npy"))

        accuracy_single = {} # key=pooling_type, value= relevance accuracy
        for pooling_type in ["sum,abs", "sum,pos", "max-norm", "l1-norm", "l2-norm", "l2-norm,sq"]:
            accuracy_single[pooling_type] = evaluate_single (heatmap, ground_truth, pooling_type) [evaluation_metric]
          
        accuracy[str(idx)] = accuracy_single
    
    # compute statistics
    accuracy_statistics = {} # key=pooling_type, value=dict containing the relevance statistics (with the statistic's name as key, and statistic's value as value)
    for pooling_type in ["sum,abs", "sum,pos", "max-norm", "l1-norm", "l2-norm", "l2-norm,sq"]:
        accuracy_statistics[pooling_type] = {}
        values = np.asarray( [ accuracy[str(idx)][pooling_type] for idx in idx_list ] ) ; assert values.shape==(len(idx_list),)
        accuracy_statistics[pooling_type]["mean"]     = np.mean(values)
        accuracy_statistics[pooling_type]["std"]      = np.std(values)
        accuracy_statistics[pooling_type]["min"]      = np.amin(values)
        accuracy_statistics[pooling_type]["max"]      = np.amax(values)
        accuracy_statistics[pooling_type]["median"]   = np.percentile(values, q=50, interpolation='linear')
        accuracy_statistics[pooling_type]["perc-80"]  = np.percentile(values, q=80, interpolation='linear')
        accuracy_statistics[pooling_type]["perc-20"]  = np.percentile(values, q=20, interpolation='linear')

    # write statistics over all pooling_types to text file in decreasing order of the mean, and formatted as a table
    list_to_sort = [(k, v["mean"]) for (k, v) in accuracy_statistics.items()]
    sorted_keys  = [ k for (k, _) in sorted(list_to_sort, key=lambda tup: tup[1], reverse=True)]
    
    col_width = 17
    with open(os.path.join(output_DIR,  output_name + "_ORDERED.txt"), "w") as stat_file:
        titles = ["pooling_type", "mean", "std", "min", "max", "median", "perc-80", "perc-20"]
        stat_file.write(''.join([(title).ljust(col_width) for title in titles]) + '\n')
        for k in sorted_keys:
            row = [k] + ["{:4.2f}".format(accuracy_statistics[k][stat_name]) for stat_name in titles[1:]]
            stat_file.write(''.join([value.ljust(col_width) for value in row]) + '\n')
        stat_file.write("\n\nRelevance accuracy metric:            " + evaluation_metric)
        stat_file.write("\n\nStatistics computed over data points: " + str(len(idx_list)))

    # write statistics over all pooling_types to text file in fixed order, and formatted as a table   
    col_width = 17
    with open(os.path.join(output_DIR,  output_name + "_FIXED.txt"), "w") as stat2_file:
        titles = ["pooling_type", "mean", "std", "median"]
        stat2_file.write(''.join([(title).ljust(col_width) for title in titles]) + '\n')
        for k in ["max-norm", "l2-norm,sq", "l2-norm", "l1-norm", "sum,abs", "sum,pos"]: # fixed order DEFINED HERE !
            row = [k] + ["{:4.2f}".format(accuracy_statistics[k][stat_name]) for stat_name in titles[1:]]
            row[2] = "("+row[2]+")" # put parenthesis around std
            stat2_file.write(''.join([value.ljust(col_width) for value in row]) + '\n')
        stat2_file.write("\n\nRelevance accuracy metric:            " + evaluation_metric)
        stat2_file.write("\n\nStatistics computed over data points: " + str(len(idx_list)))
        
    # save accuracy statistics and accuracies for each data point as JSON files
    json.dump(accuracy,            open(os.path.join(output_DIR, output_name + '_datapoint'),  "w"), indent=4)
    json.dump(accuracy_statistics, open(os.path.join(output_DIR, output_name + '_statistic'),  "w"), indent=4)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--heatmap-array-file", default=None, type=str, help="File with model performance in relation to its sparsity")
    parser.add_argument("--mask-path", default=None, type=str, help="File with model performance in relation to its sparsity")
    parser.add_argument("--single-image-path", default=None, type=str, help="File with model performance in relation to its sparsity")
    parser.add_argument("--expl-method", default="ggc", type=str, help="Chosen explainability method")
    parser.add_argument("--data-path", default=None, type=str, help="dataset path")
    parser.add_argument("--models-path", default="/shared/datasets/classification/imagenet/", type=str, help="models path")
    parser.add_argument("--pruning-iteration", default=0, type=int, help="Pruning Iteration")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--output-dir", default='.', type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    return parser

def vit_attention_map(model, image, patch_size, device):

    model(image)

    img = image
    original_w, original_h = img.shape[-2], img.shape[-1]

    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    img = img.unsqueeze(0)
    img = img.to(device)
    attention = torch.load('/scratch/expl_attention_vit.pt')
    print('Shape attenzione')
    print(attention)
    # attention = attention.unsqueeze(0)
    print(attention.shape)
    # exit()
    number_of_heads = attention.shape[1]
    attention = attention[0, :, 0, 1:].reshape(number_of_heads, -1)

    attention = attention.reshape(number_of_heads, w_featmap, h_featmap)

    # attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "nearest")[0].cpu() # originale
    attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "bilinear")[0].cpu() 
    attention = torch.sum(attention, dim=0)
    attention_of_image = nn.functional.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(original_h, original_w), mode='bilinear', align_corners=False)
    attention_of_image = attention_of_image.squeeze()
    
    return attention_of_image

def get_attention_map(model, image, patch_size, device):

    pred = model(image)
    attention = torch.load('/home/cassano/last_attention_swin.pt')
    original_image, img, original_w, original_h = get_image(image, patch_size, device)
    # original_image, img, w_featmap, h_featmap, original_w, original_h = get_image(image, patch_size, device)


    w_featmap = int(math.sqrt(attention.shape[-2])) # 8
    h_featmap = int(math.sqrt(attention.shape[-1])) # 8
    number_of_heads = attention.shape[1]

    attention_image = build_attention_image(attention, number_of_heads, w_featmap=w_featmap, h_featmap=h_featmap,original_w=original_w, original_h=original_h, patch_size=patch_size)

    return attention_image, pred
     

def build_attention_image(attention, number_of_heads, w_featmap, h_featmap, original_w, original_h, patch_size):
    number_of_heads = attention.shape[1]

    print('Original Image Shape')
    print((original_w, original_h))

    # Swin non ha il CLS quindi parto da zero
    attention = attention[0, :, 0, :].reshape(number_of_heads, -1)
    attention = attention.reshape(number_of_heads, w_featmap, h_featmap)
    print(attention.shape)
    attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "nearest")[0].cpu()
    print(attention.shape)

    attention = torch.sum(attention, dim=0)
    print(attention.shape)
    attention_of_image = nn.functional.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(original_h, original_w), mode='bilinear', align_corners=False)
    print(attention_of_image.shape)
    attention_of_image = attention_of_image.squeeze()
    print(attention_of_image.shape)
    # Normalize image_metric to the range [0, 1]
    image_metric = attention_of_image.detach().numpy()
    normalized_metric = Normalize(vmin=image_metric.min(), vmax=image_metric.max())(image_metric)

    # Apply the Reds colormap
    reds = plt.cm.Reds(normalized_metric)

    # Create the alpha channel
    alpha_max_value = 1  # Set your max alpha value original 1.0

    # Adjust this value as needed to enhance lower values visibility
    gamma = 0.5  

    # Apply gamma transformation to enhance lower values
    enhanced_metric = np.power(normalized_metric, gamma)

    # Create the alpha channel with enhanced visibility for lower values
    alpha_channel = enhanced_metric * alpha_max_value

    # Add the alpha channel to the RGB data
    rgba_mask = np.zeros((image_metric.shape[0], image_metric.shape[1], 4))
    rgba_mask[..., :3] = reds[..., :3]  # RGB
    rgba_mask[..., 3] = alpha_channel  # Alpha

    # Convert the numpy array to PIL Image
    rgba_image = Image.fromarray((rgba_mask * 255).astype(np.uint8))
    # rgba_image = (rgba_mask * 255).astype(np.uint8)
    
    return rgba_image

def get_image(original_image, patch_size, device):

    # original_image is already a tensor
    img = original_image

    transform = torchvision.transforms.ToPILImage()
    original_image = transform(original_image[0])


    (original_w, original_h) = original_image.size

    if original_image.mode !='RGB':
        raise "BW image"

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size

    img = img[:, :w, :h]

#     w_featmap = img.shape[-2] // patch_size
#     h_featmap = img.shape[-1] // patch_size

    img = img.unsqueeze(0)
    img = img.to(device)
    
    # return original_image, img, w_featmap, h_featmap, original_w, original_h
    return original_image, img, original_w, original_h

def main(args):
#    if args.output_dir:
#        utils.mkdir(args.output_dir)
#
#    utils.init_distributed_mode(args)
    print(args)

    target_labels = list(IMAGENET2012_CLASSES.items())
    device = torch.device(args.device)
    print('Device:')
    print(device)
    print(f"Resuming model from pruning iteration {args.pruning_iteration}")
    if 'vit' in args.model:
        checkpoint_path = os.path.join(args.models_path, args.model, f'vit_b_32_epoch_299_pruning_iteration_{args.pruning_iteration:02}.pth')
    else:
        checkpoint_path = os.path.join(args.models_path, args.model, f'model_epoch_89_pruning_iteration_{args.pruning_iteration:02}.pth')
    print("Model filename: " + checkpoint_path)

    if args.pruning_iteration > 0 and 'vit' in args.model:
        print('Entro nel load model')
        model = load_model(checkpoint_path, args, device=device, base_vit_interpolated=None)
        # Dentro questo model ho i pesi prunati

    if 'vit' in args.model:
        base_vit = vision_transformer.vit_b_32(pretrained=True, image_size=224)
        model = interpolate_vit(base_vit=base_vit, device=device, img_shape=(224, 224), args=args)
    else:
        model = load_model(checkpoint_path, args, num_classes=1000)
    model.eval()
    print(model)

    pooling_type =  'l2-norm,sq'# 'l1-norm' # 'max-norm' 

    if args.data_path is None:
        dataset = ImageDataset(args.single_image_path)

        for img, im_name in dataset.dataloader:
            heatmap = gradCAM(model, [model.layer4[-1]], img.unsqueeze(0), args) 
            break
        mask = Image.open(args.mask_path)
        mask = np.array(mask)
        print(evaluate_single(heatmap=heatmap, ground_truth=mask, pooling_type='l1-norm'))

    else:
        gc_ranks = []
        gc_masses = []
        ggc_ranks = []
        ggc_masses = []
        ig_ranks = []
        ig_masses = []
        at_ranks = []
        at_masses = []
        lrp_ranks = []
        lrp_masses = []
        
        skipped = 0

        imgs = []
        masks = []
        image_paths = []
        mask_paths = []
        for index, single_image_path in enumerate(glob(os.path.join(args.data_path + '*.jpg'))):
            dataset = ImageDataset(single_image_path)

            mask_path = single_image_path
            mask_path = mask_path.replace('JPEGImages', 'SegmentationClass').replace('jpg', 'png')
            
            print(dataset.dataloader)
            for img, path in dataset.dataloader:
                if os.path.exists(mask_path):
                    mask_dl = ImageDataset(mask_path)
                    imgs.append(img.unsqueeze(0))
                    mask_dl = ImageDataset(mask_path)
                    image_paths.append(path)
                    mask_paths.append(mask_path)
                    for mask, path in mask_dl.dataloader:
                        masks.append(mask.unsqueeze(0))
                        break
                else:
                    skipped += 1

                break

#             if index == 8:
#                 break

#             if index == 50:
#                 break
# 
            if index == 580:
                break

        for id, image in enumerate(tqdm(imgs)):
                if id == 101:
                    break

                if args.model == 'swin':
                    patch_size = 4
                    # image = image.to(device)
                    # model = model.to(device)
                    # heatmap_gc, pred = gradCAM(model, [model.features[-1][-1].norm1], image, args)
                    # heatmap_ggc = guided_gradCAM(model, model.features[-1][-1].norm1, image)
                    # heatmap_ig = integrated_gradients(model, image)
                    # heatmap_lrp = lrp(model, model.features[-1][-1].norm1, image) non funziona per swin
                    heatmap_attn, pred = get_attention_map(model, image, patch_size, device)

                    heatmap_attn.save(f'/home/cassano/swin_attentions/attention_{id}.png')
                    # plt.imsave(f'/home/cassano/swin_attentions/attention_{id}.jpg', heatmap_attn)
                    # heatmap_attn = heatmap_attn.squeeze().cpu().detach().numpy()
                    # print(heatmap_attn.shape)
                    # fig = plt.figure(figsize=(11, 11))
                    # attr = plt.imshow(heatmap_attn) # image.squeeze().transpose(0, 2).transpose(0, 1))
                    # plt.tight_layout()
                    # plt.imsave(f'/home/cassano/swin_attentions/attention_{id}.jpg', heatmap_attn)
                    # plt.savefig(os.path.join('/home/cassano/swin_attentions', f'{id}_original_image.jpg'))
                    # plt.close()
                elif 'vit' in args.model:
                    args.patch_size = 32
                    # Padding the image with zeros to fit multiple of patch-size
                    w_before_padding = image.shape[2]
                    h_before_padding = image.shape[3]
                    size_im = (
                        image.shape[0],
                        image.shape[1],
                        int(np.ceil(image.shape[2] / args.patch_size) * args.patch_size),
                        int(np.ceil(image.shape[3] / args.patch_size) * args.patch_size),
                    )
                    paded = torch.zeros(size_im)
                    paded[:,:, : image.shape[2], : image.shape[3]] = image
                    image = paded
                    print(image.shape)
                    
                    image = image.to(device)

#                    if args.pruning_iteration > 0 and 'vit' in args.model:
#                        print('Entro nel load model')
#                        model = load_model(checkpoint_path, args, device=device, base_vit_interpolated=None)
#                        # Dentro questo model ho i pesi prunati

                    if 'vit' in args.model:
                        model = load_model(checkpoint_path, args, device=device, base_vit_interpolated=None)
                        model = interpolate_vit(base_vit=model, device=device, img_shape=(image.shape[2], image.shape[3]), args=args)

                    model = model.to(device)

                    patch_size = 32
                    print('Image shape to attribution methods')
                    print(image.shape)
                    heatmap_gc, pred = gradCAM(model, [model.encoder], image, args, vit_img_shape=(image.shape[2]//patch_size, image.shape[3]//patch_size))
                    # heatmap_ggc = guided_gradCAM(model, model.encoder, image)
                    # heatmap_lrp = lrp(model, model.features[-1][-1].norm1, image) non funziona per swin
                    heatmap_attn = vit_attention_map(model, image, patch_size, device)
                    heatmap_ig = integrated_gradients(model, image)

                else:
                    # image = image.to(device)
                    # model = model.to(device)
                    heatmap_gc, pred = gradCAM(model, [model.layer4[-1]], image, args) 
                    heatmap_ggc = guided_gradCAM(model, model.layer4, image) 
                    heatmap_ig = integrated_gradients(model, image)
                    # heatmap_lrp = lrp(model, image) 

                # Denormalization: necessaria per aggiustare il contrasto delle immagini caricate con il dataloader.
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])

                normalize = transforms.Normalize(mean.tolist(), std.tolist()) 

                unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                np_original_image = unnormalize(image)
                np_original_image = np_original_image.squeeze().cpu().detach().numpy()
                fig = plt.figure(figsize=(11, 11))
                attr = plt.imshow(np.transpose(np_original_image, (1, 2, 0))) # image.squeeze().transpose(0, 2).transpose(0, 1))
                plt.title(f'Path: {image_paths[id]}')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'ex_eval_imgs', f'{id}_original_image.jpg'))
                # plt.savefig(os.path.join('/home/cassano/swin_attentions', f'{id}_original_image.jpg'))
                plt.close()


                metrics_gc, gt = evaluate_single(heatmap=heatmap_gc, ground_truth=masks[id][0][0].detach().numpy(), pooling_type=None)
                gc_masses.append(metrics_gc['mass'])
                gc_ranks.append(metrics_gc['rank'])

                fig = plt.figure(figsize=(11, 11))
                attr = plt.imshow(heatmap_gc.squeeze(), cmap='jet')
                plt.colorbar(attr, location='bottom') # cmap=cv2.COLORMAP_JET)
                plt.title(f'Mass: {metrics_gc["mass"]}, Rank: {metrics_gc["rank"]}, Pred: {target_labels[np.argmax(pred)][1]}')
                plt.tight_layout()
                dir = os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                plt.savefig(os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}' , f'{id}_heatmap_gc.jpg'))
                plt.close()

                # hggc = heatmap_ggc[0].squeeze().cpu().detach().numpy()
                # metrics_ggc, _ = evaluate_single(heatmap=hggc, ground_truth=masks[id][0][0].detach().numpy(), pooling_type=pooling_type)
                # ggc_masses.append(metrics_ggc['mass'])
                # ggc_ranks.append(metrics_ggc['rank'])

                # figure, axis = viz.visualize_image_attr(np.transpose(hggc, (1,2,0)),
                #                      np.transpose(np_original_image, (1,2,0)),
                #                      method='heat_map',
                #                      # cmap=default_cmap,
                #                      show_colorbar=True,
                #                      sign='all',
                #                      outlier_perc=1, 
                #                      title = f'Mass: {metrics_ggc["mass"]}, Rank: {metrics_ggc["rank"]}, Pred: {target_labels[np.argmax(pred)][1]}'
                #                      )
                # plt.tight_layout()
                # dir = os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}')
                # if not os.path.exists(dir):
                #     os.mkdir(dir)
                # plt.savefig(os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}',   f'{id}_heatmap_ggc.jpg'))
                # plt.close()

                fig = plt.figure(figsize=(11, 11))
                attr = plt.imshow(gt)
                plt.title(f'Path: {mask_paths[id]}')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'ex_eval_imgs',  f'{id}_ground_truth_mask.jpg'))
                plt.close()

                print(heatmap_attn.shape)
                hat = heatmap_attn.squeeze().cpu().detach().numpy()

                hat = np.transpose(hat)
                print(heatmap_attn.shape)
                # exit()

                metrics_at, _ = evaluate_single(heatmap=hat, ground_truth=masks[id][0][0].detach().numpy(), pooling_type=None)
                at_masses.append(metrics_at['mass'])
                at_ranks.append(metrics_at['rank'])

                # image_metric = hat # [:np_original_image.shape[0], :np_original_image.shape[1]]
                
                print(hat.shape)
                # image_metric = hat.resize(hat, (hat.shape[0], w_before_padding, h_before_padding)) # [:np_original_image.shape[0], :np_original_image.shape[1]]

                image_metric = hat
                normalized_metric = Normalize(vmin=image_metric.min(), vmax=image_metric.max())(image_metric)
        
                # Apply the Reds colormap
                reds = plt.cm.Reds(normalized_metric)
        
                # Create the alpha channel
                alpha_max_value = 1.00  # Set your max alpha value
        
                # Adjust this value as needed to enhance lower values visibility
                gamma = 0.5  
        
                # Apply gamma transformation to enhance lower values
                enhanced_metric = np.power(normalized_metric, gamma)
        
                # Create the alpha channel with enhanced visibility for lower values
                alpha_channel = enhanced_metric * alpha_max_value
        
                # Add the alpha channel to the RGB data
                rgba_mask = np.zeros((image_metric.shape[0], image_metric.shape[1], 4))
                rgba_mask[..., :3] = reds[..., :3]  # RGB
                rgba_mask[..., 3] = alpha_channel  # Alpha
        
                # Convert the numpy array to PIL Image
                rgba_image = Image.fromarray((rgba_mask * 255).astype(np.uint8))

        
                # Save the image
                dir = os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                rgba_image.save(os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}' ,  f'{id}_heatmap_attention.png'))
                # rgba_image.save(os.path.join(args.'attention_mask.png')
                fig = plt.figure(figsize=(11, 11))
                plt.imshow(np.transpose(np_original_image[:, :w_before_padding, :h_before_padding], (1, 2, 0)), alpha=.75)
                attr = plt.imshow(rgba_image,  alpha=0.95, cmap='jet')
                plt.colorbar(attr, location='bottom') # cmap=cv2.COLORMAP_JET)
                # plt.imshow(rgba_mask, alpha=.90)
                plt.title(f'Mass: {metrics_at["mass"]}, Rank: {metrics_at["rank"]}, Pred: {target_labels[np.argmax(pred)][1]}')
                plt.tight_layout()
                dir = os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                plt.savefig(os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}' ,  f'{id}_blended_heatmap_attention.jpg'))
                plt.close()



                print(heatmap_ig.shape)
                hig = heatmap_ig[0].squeeze().cpu().detach().numpy()
                print(heatmap_ig.shape)
                # exit()

                metrics_ig, _ = evaluate_single(heatmap=hig, ground_truth=masks[id][0][0].detach().numpy(), pooling_type=pooling_type)
                ig_masses.append(metrics_ig['mass'])
                ig_ranks.append(metrics_ig['rank'])

                figure, axis = viz.visualize_image_attr(np.transpose(hig, (1,2,0)),
                                     np.transpose(np_original_image, (1,2,0)),
                                     method='heat_map',
                                     # cmap=default_cmap,
                                     show_colorbar=True,
                                     sign='all',
                                     outlier_perc=1, 
                                     title = f'Mass: {metrics_ig["mass"]}, Rank: {metrics_ig["rank"]}, Pred: {target_labels[np.argmax(pred)][1]}'
                                     )
                plt.tight_layout()
                dir = os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                plt.savefig(os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}' ,  f'{id}_heatmap_ig.jpg'))
                plt.close()

                # if args.model != 'swin':
                if False:
                    hlrp = heatmap_lrp[0].squeeze().cpu().detach().numpy()
                    metrics_lrp, _ = evaluate_single(heatmap=hlrp, ground_truth=masks[id][0][0].detach().numpy(), pooling_type=pooling_type)
                    lrp_masses.append(metrics_lrp['mass'])
                    lrp_ranks.append(metrics_lrp['rank'])

                    figure, axis = viz.visualize_image_attr(np.transpose(hlrp, (1,2,0)),
                                         np.transpose(np_original_image, (1,2,0)),
                                         method='heat_map',
                                         # cmap=default_cmap,
                                         show_colorbar=True,
                                         sign='all',
                                         outlier_perc=1, 
                                         title = f'Mass: {metrics_lrp["mass"]}, Rank: {metrics_lrp["rank"]}, Pred: {target_labels[np.argmax(pred)][1]}'
                                         )
                    plt.tight_layout()
                    dir = os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}')
                    if not os.path.exists(dir):
                        os.mkdir(dir)
                    plt.savefig(os.path.join(args.output_dir,  'ex_eval_imgs', f'{args.model}', f'pruning_iteration_{args.pruning_iteration:02}' , f'{id}_heatmap_lrp.jpg'))
                    plt.close()



        gc_avg_mass = sum(gc_masses)/(index-skipped+1)
        gc_avg_rank = sum(gc_ranks)/(index-skipped+1)
        at_avg_mass = sum(at_masses)/(index-skipped+1)
        at_avg_rank = sum(at_ranks)/(index-skipped+1)
        # ggc_avg_mass = sum(ggc_masses)/(index-skipped+1)
        # ggc_avg_rank = sum(ggc_ranks)/(index-skipped+1)
        ig_avg_mass = sum(ig_masses)/(index-skipped+1)
        ig_avg_rank = sum(ig_ranks)/(index-skipped+1)

        # if args.model != 'swin':
        if False:
            lrp_avg_mass = sum(lrp_masses)/(index-skipped+1)
            lrp_avg_rank = sum(lrp_ranks)/(index-skipped+1)
            with open(os.path.join(f'{args.output_dir}', f'lrp_{args.model}_pruning_iteration_{args.pruning_iteration:02}.txt'), 'w') as f:
                f.write("Average Mass: " + str(lrp_avg_mass) + '\n')
                f.write("Average Rank: " + str(lrp_avg_rank) + '\n')
                f.write('On a total of ' + str(index-skipped+1) + ' images.' + '\n')

        with open(os.path.join(f'{args.output_dir}', f'gradCAM_{args.model}_pruning_iteration_{args.pruning_iteration:02}.txt'), 'w') as f:
            f.write("Average Mass: " + str(gc_avg_mass) + '\n')
            f.write("Average Rank: " + str(gc_avg_rank) + '\n')
            f.write('On a total of ' + str(index-skipped+1) + ' images.' + '\n')

        with open(os.path.join(f'{args.output_dir}', f'attention_{args.model}_pruning_iteration_{args.pruning_iteration:02}.txt'), 'w') as f:
            f.write("Average Mass: " + str(at_avg_mass) + '\n')
            f.write("Average Rank: " + str(at_avg_rank) + '\n')
            f.write('On a total of ' + str(index-skipped+1) + ' images.' + '\n')

#           with open(os.path.join(f'{args.output_dir}', f'guided_gradCAM_{args.model}_pruning_iteration_{args.pruning_iteration:02}.txt'), 'w') as f:
#               f.write("Average Mass: " + str(ggc_avg_mass) + '\n')
#               f.write("Average Rank: " + str(ggc_avg_rank) + '\n')
#               f.write('On a total of ' + str(index-skipped+1) + ' images.' + '\n')
#   
        with open(os.path.join(f'{args.output_dir}', f'ig_{args.model}_pruning_iteration_{args.pruning_iteration:02}.txt'), 'w') as f:
            f.write("Average Mass: " + str(ig_avg_mass) + '\n')
            f.write("Average Rank: " + str(ig_avg_rank) + '\n')
            f.write('On a total of ' + str(index-skipped+1) + ' images.' + '\n')

    print('End.')

def guided_gradCAM(model, layer, image_batch):
    guided_gc = GuidedGradCam(model, layer)

    pred = model(image_batch)
    pred = pred.cpu().detach().numpy()
    predicted_class_numbers = []
    for p in pred:
        predicted_class_numbers.append(np.argmax(p))

    predicted_class_numbers = torch.tensor(predicted_class_numbers)
    attribution = guided_gc.attribute(image_batch, predicted_class_numbers)

    return attribution

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(3))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def gradCAM(model, layer, image_batch, args, vit_img_shape=None):
    if args.model == 'swin':
        cam = GradCAM(model=model, target_layers=layer, reshape_transform=reshape_transform)
    elif 'vit' in args.model:
        if vit_img_shape is None:
            # cam = GradCAM(model=model, target_layers=layer, reshape_transform=vit_reshape_transform)
            pass
        else:
            cam = GradCAM(model=model, target_layers=layer, reshape_transform=lambda tensor, height=vit_img_shape[0], width=vit_img_shape[1]: tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2)).transpose(2, 3).transpose(1, 2))
    else:
        cam = GradCAM(model=model, target_layers=layer)

    pred = model(image_batch)
    pred = pred.cpu().detach().numpy()

    # Required By GradCAM
    new_targets = []
    for p in pred:
        new_targets.append(ClassifierOutputTarget(np.argmax(p)))

    attribution = cam(input_tensor=image_batch, targets=new_targets) #  , eigen_smooth=True, aug_smooth=True)
    return attribution, pred[0]

def lrp(model, image_batch):
    lrp = LRP(model)

    pred = model(image_batch)
    pred = pred.cpu().detach().numpy()

    predicted_class_numbers = []
    for p in pred:
        predicted_class_numbers.append(np.argmax(p))

    predicted_class_numbers = torch.tensor(predicted_class_numbers)
    attribution = lrp.attribute(image_batch, predicted_class_numbers)

    return attribution

def integrated_gradients(model, image_batch): 
    ig = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(ig)

    pred = model(image_batch)
    pred = pred.cpu().detach().numpy()

    predicted_class_numbers = []
    for p in pred:
        predicted_class_numbers.append(np.argmax(p))

    predicted_class_numbers = torch.tensor(predicted_class_numbers)
    attribution = noise_tunnel.attribute(nt_samples_batch_size = 1, inputs=image_batch, target=predicted_class_numbers, nt_samples=2, nt_type='smoothgrad')

    return attribution



def get_explainations(image, target=None, model=None, args=None, pruning_iteration=None, target_labels=None, batch_index=None):
            if args.model == 'swin':
                guided_gradCAM(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index)
                gradCAM(model, [model.features[-1][-1].norm1], image, target, args, pruning_iteration, target_labels, batch_index)
                # lrp(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index)
                integrated_gradients(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index)
            else:
                guided_gradCAM(model, [model.layer4[-1]], image) 
                gradCAM(model, [model.layer4[-1]], image) 
                lrp(model, image)
                integrated_gradients(model, image)

# def load_model(checkpoint_path, args, num_classes):
#     if args.model == 'swin':
#         model = torchvision.models.swin_v2_t()
#         pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
#     else:
#         model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
#         if args.model == 'resnet50':
#             pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
#         else:
#             pruned_layer_types = [torch.nn.Conv2d]
# 
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')["model"]
# 
#     correct_checkpoint = OrderedDict()
#     for k in checkpoint:
#         correct_checkpoint[k.replace("module.", "")] = checkpoint[k]
# 
#     try:
#         model.load_state_dict(correct_checkpoint)
#     except:
#         for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
#             prune.identity(module, "weight")
# 
#         try:
#             model.load_state_dict(correct_checkpoint)
# 
#             for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
#                 prune.remove(module, "weight")
#         except:
#             raise RuntimeError()
# 
#     return model

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)