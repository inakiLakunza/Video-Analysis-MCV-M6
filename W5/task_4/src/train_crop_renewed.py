""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from datasets import HMDB51Dataset, TSNHMDB51Dataset, FiveCropTSNHMDB51Dataset_renewed
from models import model_creator
from utils import model_analysis
from utils import statistics

import wandb
import os

from callbacks import EarlyStopper, SaveBestModel



def train(
        model: nn.Module,
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module,
        device: str,
        epoch:int,
        description: str = ""
    ) -> None:
    """
    Trains the given model using the provided data loader, optimizer, and loss function.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader containing the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        loss_fn (nn.Module): The loss function used to compute the training loss.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.train()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    cont = (len(train_loader)) * epoch + 1 if epoch is not None else 0

    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass
        outputs = model(clips).view(labels.shape[0], clips.shape[0] // labels.shape[0], -1 )
        outputs_aggregated = torch.mean(outputs, dim=1)
        # Compute loss
        loss = loss_fn(outputs_aggregated, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update progress bar with metrics
        loss_iter = loss.item()
        hits_iter = torch.eq(outputs_aggregated.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        mean_loss = loss_train_mean(loss_iter)

        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )
        
        if epoch:
            wandb.log({"epoch": epoch+1, 
                       "step": cont+1, 
                       "train_running_loss": loss_train_mean(loss_iter),
                       "train_loss": loss_iter, 
                       "train_acc": (float(hits_iter) / len(labels))}
                    )
        cont += 1
    
    if epoch: wandb.log({"epoch": epoch+1, "epoch_avg_train_loss": mean_loss, "epoch_avg_train_acc": (float(hits) / count)})
    
    return mean_loss


def evaluate(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        device: str,
        epoch:int,
        description: str = ""
    ) -> None:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    cont = (len(valid_loader)) * epoch + 1 if epoch is not None else 0
    mean_loss = 0
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)

        # Forward pass
        with torch.no_grad():
            #clips = clips.view(-1, 3, 1, clips.shape[3], clips.shape[4])
            outputs = model(clips).view(labels.shape[0], clips.shape[0] // labels.shape[0], -1 )
            outputs_aggregated = torch.mean(outputs, dim=1)
            
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
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )


            if epoch:
                wandb.log({"epoch": epoch+1, 
                           "step": cont+1, 
                           "val_running_loss": loss_valid_mean(loss_iter),
                           "val_loss": loss_iter, 
                           "val_acc": (float(hits_iter) / len(labels))}
                        )

            cont += 1
            
    # Mean loss for current epoch
    if epoch: wandb.log({"epoch": epoch+1, "epoch_avg_val_loss": mean_loss, "epoch_avg_val_acc": (float(hits) / count)})
    
    if epoch is None: epoch = 1
    early_stopper.early_stop(mean_loss, model, optimizer, epoch)

    return mean_loss


def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        n_segments: int
) -> Dict[str, TSNHMDB51Dataset]:
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
    for regime in FiveCropTSNHMDB51Dataset_renewed.Regime:
        datasets[regime.name.lower()] = FiveCropTSNHMDB51Dataset_renewed(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride,
            n_segments
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


def create_optimizer(optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    Creates an optimizer for the given parameters.
    
    Args:
        optimizer_name (str): Name of the optimizer (supported: "adam" and "sgd" for now).
        parameters (Iterator[nn.Parameter]): Iterator over model parameters.
        lr (float, optional): Learning rate. Defaults to 1e-4.

    Returns:
        torch.optim.Optimizer: The optimizer for the model parameters.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def print_model_summary(
        model: nn.Module,
        clip_length: int,
        crop_size: int,
        print_model: bool = True,
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
    parser.add_argument('--frames_dir', type=str, 
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
    
    parser.add_argument('--load-pretrain', action='store_true', default=False,
                    help='Load pretrained weights for the model (if available)')
    parser.add_argument('--optimizer-name', type=str, default="adam",
                        help='Optimizer name (supported: "adam" and "sgd" for now)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=16,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--validate-every', type=int, default=5,
                        help='Number of epochs after which to validate the model')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')

    args = parser.parse_args()
    
    
    wandb.login(key="34db2c5ef8832f040bb5001755f4aa5b64cf78fa",
                relogin=True)
    
    wandb.init(
        project = "C6-W5",
        name = "train_task_d",
        config={
            "tokenizer": "character-level",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "batch_size_eval": args.batch_size_eval,
            "optimizer": args.optimizer_name,
            "learning_rate": args.lr
        }
    )

    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("val_loss", summary="min")

    wandb.define_metric("train_acc", summary="max")
    wandb.define_metric("val_acc", summary="max")
    
    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=FiveCropTSNHMDB51Dataset_renewed.Split.TEST_ON_SPLIT_1, # hardcoded
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

    # Init model, optimizer, and loss function
    model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print_model_summary(model, args.clip_length, args.crop_size)

    model = model.to(args.device)
    #wandb.watch(model, log_freq=100)


    # CALLBACKS
    #==========================================
    # SAVE BEST MODEL
    SAVE_FOLDER = "./weights"
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    SAVE_NAME = "task_d_300_epochs_average_votting_3seg_8_clip.pth"
    save_path = os.path.join(SAVE_FOLDER, SAVE_NAME)
    print("Best model will be saved in the following path:\n", save_path)
    save_best_model = SaveBestModel(save_path)

    # EARLY STOPPING
    early_stopper = EarlyStopper(save_path, patience=15, min_delta=0.)
    #==========================================

    for epoch in range(args.epochs):
        # Validation
        if epoch % args.validate_every == 0:
            description = f"Validation [Epoch: {epoch+1}/{args.epochs}]"
             # Preguntar Demà
            val_loss = evaluate(model, loaders['validation'], loss_fn, args.device, epoch=epoch, description=description)
            # Stop if patience is ended
            if early_stopper.get_stop(): 
                break       
       
        # Training
        description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
        train(model, loaders['training'], optimizer, loss_fn, args.device, epoch=epoch, description=description)


    # SAVE WEIGHTS OF THE BEST MODEL
    save_best_model.save(val_loss, model, optimizer, args.epochs)



    # Testing
    evaluate(model, loaders['validation'], loss_fn, args.device, description=f"Validation [Final]")
    evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing")

    exit()
