""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator, Tuple

from torch.utils.data import DataLoader

from datasets import HMDB51Dataset, TSNHMDB51Dataset
from models import model_creator
from utils import model_analysis
from utils import statistics


import os


import wandb

from collections import defaultdict

def evaluate(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        device: str,
        description: str = ""
    ) -> Tuple[float, Dict[str, float], float]:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        Tuple[float, Dict[str, float], float]: Overall accuracy, Accuracy per class, Mean loss
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0  # auxiliary variables for computing accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        skeleton = batch["skeleton"].to(device)
        clips_motion, skeleton_motion = batch["clips_motion"].to(device), batch["skeleton_motion"].to(device)
        magnitude_skeleton_motion = batch["skeleton_motion_magnitude"].to(device)
        
        
        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            magnitud_outputs = model(clips_motion)
            
            outputs_aggregated = magnitud_outputs#(outputs + magnitud_outputs)  #torch.mean(outputs, dim=1)
            
            
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs_aggregated, labels) 

            # Compute metrics
            loss_iter = loss.item()
            hits_iter = torch.eq(outputs_aggregated.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)
            # Update progress bar with metrics
            mean_loss = loss_valid_mean(loss_iter)

            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=mean_loss,
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )
            
            # Update class-wise accuracy
            for pred, true_label in zip(outputs_aggregated.argmax(dim=1), labels):
                class_correct[true_label.item()] += int(pred == true_label)
                class_total[true_label.item()] += 1

    # Calculate accuracy per class
    accuracy_per_class = {class_idx: class_correct[class_idx] / class_total[class_idx] for class_idx in class_correct}
            
    return (float(hits) / count), accuracy_per_class, loss_valid_mean(loss_iter)



def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        n_segments: int
) -> Dict[str, HMDB51Dataset]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.

    Returns:
        Dict[str, HMDB51Dataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    
    #### Aquñi Hardcoded the dataset
    for regime in HMDB51Dataset.Regime:
        datasets[regime.name.lower()] = HMDB51Dataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride
            #n_segments
        )
    
    return datasets


def create_dataloaders(
        datasets: Dict[str, HMDB51Dataset],
        batch_size: int,
        batch_size_eval: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        datasets (Dict[str, HMDB51Dataset]): A dictionary containing datasets for training, validation, and testing.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfer. Defaults to True.

    Returns:
        Dict[str, DataLoader]: A dictionary containing data loaders for training, validation, and testing datasets.
    """
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=(batch_size if key == 'training' else batch_size_eval),
            shuffle=(key == 'training'),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
            
    return dataloaders


def print_model_summary(
        model: nn.Module,
        clip_length: int,
        crop_size: int,
        print_model: bool = False,
        print_params: bool = True,
        print_FLOPs: bool = True
    ) -> None:
    """
    Prints a summary of the given model.

    Args:
        model (nn.Module): The model for which to print the summary.
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        print_model (bool, optional): Whether to print the model architecture. Defaults to True.
        print_params (bool, optional): Whether to print the number of parameters. Defaults to True.
        print_FLOPs (bool, optional): Whether to print the number of FLOPs. Defaults to True.

    Returns:
        None
    """
    if print_model:
        print(model)

    if print_params:
        num_params = sum(p.numel() for p in model.parameters())
        #num_params = model_analysis.calculate_parameters(model) # should be equivalent
        print(f"Number of parameters (M): {round(num_params / 10e6, 2)}")

    if print_FLOPs:
        num_FLOPs = model_analysis.calculate_operations(model, clip_length, crop_size, crop_size)
        print(f"Number of FLOPs (G): {round(num_FLOPs / 10e9, 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('--frames_dir', default="/home/cboned/Desktop/Master/mcv-c6-2024-team6/data/frames", type=str, 
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    
    parser.add_argument('--clip-length', type=int, default=4,
                        help='Number of frames of the clips')
    
    parser.add_argument('--crop-size', type=int, default=182,
                        help='Size of spatial crops (squares)')
    
    parser.add_argument('--temporal-stride', type=int, default=12,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    
    parser.add_argument("--n_segments", type=int, default=6,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS'
                        )
    
    parser.add_argument('--model-name', type=str, default='x3d_xs',
                        help='Model name as defined in models/model_creator.py')
    
    parser.add_argument('--load-pretrain', action='store_true', default=True,
                    help='Load pretrained weights for the model (if available)')
    
    parser.add_argument('--optimizer-name', type=str, default="adam",
                        help='Optimizer name (supported: "adam" and "sgd" for now)')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for the training data loader')
    
    parser.add_argument('--batch-size-eval', type=int, default=8,
                        help='Batch size for the evaluation data loader')
    
    parser.add_argument('--validate-every', type=int, default=10,
                        help='Number of epochs after which to validate the model')
    
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    
    parser.add_argument('--strategy', type=str, default='difference',
                        help='Test different strategy')
    

    args = parser.parse_args()
    
    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
        n_segments = args.n_segments
    )

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )
    
    MODEL_WIGHTS_FOLDER = "./weights"
    RESULTS_PATH = "./results" 

    template = "epoch_40_x3d_xs_difference_difference_rgb_finetunning"
    
    file_to_save_results = template + ".txt"
    model_name = template + ".pth"

    # Init model, optimizer, and loss function
    model, params_to_train = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())

    print_model_summary(model, args.clip_length, args.crop_size)

    model.load_state_dict(torch.load(os.path.join(MODEL_WIGHTS_FOLDER, model_name)))
    model = model.to(args.device)
    
    print("EVALUATING ON MODEL :", model_name)
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.3)

    # Testing
    final_val_acc, acc_per_class, final_val_loss = evaluate(model, loaders['validation'], loss_fn=loss_fn, device=args.device, description=f"Validation [Final]")
    print(f"FINAL VALIDATION ACC: {final_val_acc} FINAL VALIDATION LOSS: {final_val_loss}")
    final_test_acc, acc_per_class,  final_test_loss = evaluate(model, loaders['testing'], loss_fn=loss_fn, device=args.device, description=f"Testing")
    print(f"FINAL TEST ACC: {final_test_acc} FINAL TEST LOSS: {final_test_loss}")


    with open(os.path.join(RESULTS_PATH, file_to_save_results), 'a') as file:
        file.write(f"Overall Accuracy: {final_test_acc}\n")
        file.write("Accuracy per class:\n")
        file.write(f"{acc_per_class}\n")
        file.write(f"Mean Loss: {final_test_loss}\n\n")

    exit()
