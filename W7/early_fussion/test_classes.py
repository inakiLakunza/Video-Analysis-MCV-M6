""" Main script for training a video classification model on HMDB51 dataset. """

import os
import sys
sys.path.append('./..')

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator

from torch.utils.data import DataLoader

from src.datasets.HMDB51Dataset import HMDB51Dataset
from src.models import model_creator
from src.utils import model_analysis
from src.utils import statistics


from callbacks import EarlyStopper, SaveBestModel

from torchvision.models.optical_flow import raft_small

import build_model




def evaluate(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        device: str,
        epoch: int, 
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
    acc_class_dict = {}
    class_cont = {}
    pred_labels = []
    true_labels = []
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        clips_og = clips[:, :, :-1, :, :] # drop last clip
        # clips is (16, 3, 9, 224, 224) and dtype torch.float
        # I removed the normalization with the mean and std from the dataloader!

        # Compute Optical Flow with RAFT =========================================
        opfs = []
        for i in range(0, clips.shape[-3] - 1):
            # opf is computed with 2 consecutive clips
            clip1 = clips[:, :, i, :, :]
            clip2 = clips[:, :, i+1, :, :]
            flow = raft_model(clip1, clip2)[0]
            # print(f'Flow {i} shape {flow.shape}')
            opfs.append(flow)
        of = torch.stack(opfs).permute(1, 2, 0, 3, 4) # (16, 2, 8, 224, 224)
        # ========================================================================

        # CONCATENATE RGB AND OF =========================================================================
        input_info = torch.concat([clips_og, of], dim=1)
        #=========================================================================


        with torch.no_grad():
            outputs = model(input_info)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels) 
            # Compute metrics
            loss_iter = loss.item()
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)
            # Update progress bar with metrics
            mean_loss = loss_valid_mean(loss_iter)

            pred = outputs.argmax(dim=1)
            pred_labels.extend(pred.tolist())
            true_labels.extend(labels.tolist())


            acc_class_dict[labels.item()] = acc_class_dict.get(labels.item(), 0) + (float(hits_iter) / len(labels))
            class_cont[labels.item()] = class_cont.get(labels.item(), 0) + 1

            # print("float(hits_iter): ", float(hits_iter))
            # print("len(labels): ", len(labels))
            # print("their division: ", float(hits_iter) / len(labels))


            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )

            cont += 1

    results = {}
    counts = {}
    for k in acc_class_dict.keys():
        results[CLASS_NAMES[k]] = acc_class_dict[k] / class_cont[k]
        counts[CLASS_NAMES[k]] = class_cont[k]
        # print(f'KEY: {CLASS_NAMES[k]} ACC = {acc_class_dict[k] / class_cont[k]}')

    print(f'Full Dict Results ========================================\n{results}')
    print(f'Full Dict Counts ========================================\n{counts}')
    print(f'Pred Labels ========================================\n{pred_labels}')
    print(f'True Labels ========================================\n{true_labels}')



def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int
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
    for regime in HMDB51Dataset.Regime:
        datasets[regime.name.lower()] = HMDB51Dataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride
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
    parser.add_argument('--frames_dir', type=str, default="/ghome/group07/test/W5/frames/",
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="/ghome/group07/test/W5/data/hmdb51/testTrainMulti_601030_splits/",
                        help='Directory containing annotation files')
    parser.add_argument('--clip-length', type=int, default=9,
                        help='Number of frames of the clips')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size of spatial crops (squares)')
    parser.add_argument('--temporal-stride', type=int, default=12,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
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
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=1,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--validate-every', type=int, default=1,
                        help='Number of epochs after which to validate the model')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')

    args = parser.parse_args()


    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride
    )

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )


    CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]


    # LOAD MODEL
    
    # MODIFIED RESNET ==================================================================================================
    load_path = "/ghome/group07/test/W7/early_fusion/weights/modified_resnet_raft_resnet_pretrained_all_layers_no_freeze.pth"

    model = build_model.create('modified_resnet', args.load_pretrain, datasets["training"].get_num_classes(), 
                               new_in_channels=5)
    #==========================================================================================================

    # RESNET ==================================================================================================
    #model = build_model.create('3d_resnet', args.load_pretrain, datasets["training"].get_num_classes(), 
    #                           new_in_channels=5)
    #Resnet Freezing
    #load_path = "/ghome/group07/test/W7/early_fusion/weights/resnet_raft_resnet_pretrained_all_layers.pth"

    # Resnet fine tune all
    #load_path = "/ghome/group07/test/W7/early_fusion/weights/resnet_raft_resnet_pretrained_all_layers_no_freeze.pth"
    #==========================================================================================================


    # X3D XS ==================================================================================================
    #model = build_model.create('x3d_xs', args.load_pretrain, datasets["training"].get_num_classes(), 
    #                            new_in_channels=5)
    
    # X3D fine tune all
    #load_path = "/ghome/group07/test/W7/early_fusion/weights/x3d_raft_resnet_pretrained_all_layers_no_freeze.pth"
    
    # X3D freezing
    #load_path = "/ghome/group07/test/W7/early_fusion/weights/x3d_raft_pretrained_all_layers.pth"
    #load_path = "/ghome/group07/test/W7/early_fusion/weights/x3d_raft_resnet_pretrained_all_layers.pth.pth"
    #==========================================================================================================
    


    # Init model, optimizer, and loss function
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Init RAFT OPF
    raft_model = raft_small(pretrained=True, progress=False).to(args.device)

    #print_model_summary(model, args.clip_length, args.crop_size)

    model = model.to(args.device)

    




    checkpoint = torch.load(load_path) 
    model.load_state_dict(checkpoint['model_state_dict']) 
    model = model.to(args.device)
    # Testing
    evaluate(model, loaders['testing'], loss_fn, args.device, None, description=f"Testing")

    exit()
