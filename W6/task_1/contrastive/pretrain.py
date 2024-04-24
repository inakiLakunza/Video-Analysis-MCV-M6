""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Iterator, Type

from torch.utils.data import DataLoader

from datasets import HMDB51Dataset, TSNHMDB51Dataset
from models.feature_extractor_layers import *
from models.recurrent_model import *
from models.model_creator import *

from utils import model_analysis
from utils import statistics

from torch.utils.data import Dataset


from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import os 

from torchvision.transforms import v2
from callbacks import EarlyStopper, SaveBestModel

import wandb

import numpy as np

import pipes 



device = "cuda" if torch.cuda.is_available() else "cpu"


def pretrain(model: nn.Module,
             train_loader: DataLoader,
             optimizer:torch.optim.Optimizer,
             epoch):
    
    
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    
    reducer = reducers.ThresholdReducer(low=0)
    loss_fn = losses.TripletMarginLoss(margin=0.2, reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")

    cont = (len(train_loader)) * epoch + 1 if epoch is not None else 0

    for batch in pbar:
        optimizer.zero_grad()

        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass
        outputs = model(clips)
        indices_tuple = mining_func(outputs, labels)

        # Compute loss
        loss = loss_fn(outputs, labels, indices_tuple)
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update progress bar with metrics
        loss_iter = loss.item()
        mean_loss = loss_train_mean(loss_iter)

        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter)
                            )
        
        if epoch:
            wandb.log({"epoch": epoch+1, 
                        "step": cont+1, 
                        "val_running_loss": loss_train_mean(loss_iter), 
                    })

        cont += 1
    
    if epoch: wandb.log({"epoch": epoch+1, "epoch_avg_train_loss": mean_loss})


### convenient function from pytorch-metric-learning ###


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
@torch.no_grad()
def test(train_set, test_set, model, accuracy_calculator, epoch, batch_size):

    transforms = v2.Compose([
                v2.CenterCrop(args.crop_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    model.eval()
    print("training_extraction")
    train_embeddings, train_labels = pipes.get_all_embeddings(train_set, model, transforms)
    print("testing_extraction")
    test_embeddings, test_labels = pipes.get_all_embeddings(test_set, model, transforms)
    print("Computing accuracy")

    distances = torch.cdist(test_embeddings, train_embeddings)

    k_values = [1, 3, 5, 10]
    precisions = []

    for k in k_values:
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)

        nearest_neighbor_labels = train_labels[indices]
        predicted_labels = nearest_neighbor_labels[:, :k]

        correct = torch.sum(predicted_labels == test_labels.view(-1, 1).expand_as(predicted_labels), dim=1)
        precision = (torch.sum(correct.float() > 0) / test_labels.shape[0]).item()

        print("Test set accuracy (Precision@{}) = {}".format(k, precision))
        precisions.append(precision)

    wandb.log({"epoch": epoch+1, "val_acc@1": precisions[0], "val_acc@3": precisions[1], 
               "val_acc@5": precisions[2], "val_acc@10": precisions[3]})

    return precisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('--frames_dir', type=str, default="../../frames",
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="../../data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    parser.add_argument('--clip_length', type=int, default=4,
                        help='Number of frames of the clips')
    parser.add_argument('--crop_size', type=int, default=182,
                        help='Size of spatial crops (squares)')
    
    parser.add_argument('--n_segments', type=int, default=4,
                        help='Segments in which to divide the videos')

    parser.add_argument('--temporal-stride', type=int, default=12,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    parser.add_argument('--model_name', type=str, default='x3d_xs',
                        help='Model name as defined in models/model_creator.py')
    parser.add_argument('--load-pretrain', action='store_true', default=False,
                    help='Load pretrained weights for the model (if available)')
    parser.add_argument('--optimizer-name', type=str, default="adam",
                        help='Optimizer name (supported: "adam" and "sgd" for now)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
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
        name = f"Contrastive pretrain 1 blocks {args.model_name} Triple Miner epochs=100 early=15",
        config={
            "tokenizer": "character-level",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "batch_size_eval": args.batch_size_eval,
            "optimizer": args.optimizer_name,
            "learning_rate": args.lr
        }
    )

    # Create datasets
    datasets = pipes.create_datasets(dataset=TSNHMDB51Dataset,
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
        n_segments = args.n_segments
    )
    
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("val_loss", summary="min")

    wandb.define_metric("train_acc", summary="max")
    wandb.define_metric("val_acc", summary="max")

    # Create data loaders
    loaders = pipes.create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    # Init model, optimizer, and loss function
    #model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())
    
    model = ImageFeatureExtractor3D(name_model=args.model_name, freeze=True)

    params_n = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_n.append(param)  

    optimizer = pipes.create_optimizer(args.optimizer_name, params_n, lr=args.lr)

    pipes.print_model_summary(model, args.clip_length, args.crop_size, print_model=False, params=params_n)
    model = model.to(args.device)
    wandb.watch(model, log_freq=100)
    
    
        # CALLBACKS
    #==========================================
    # SAVE BEST MODEL
    SAVE_FOLDER = "./weights"
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    SAVE_NAME = f"pretrain_3d_{args.model_name}_freeze_constrastive.pth"
    save_path = os.path.join(SAVE_FOLDER, SAVE_NAME)
    print("Best model will be saved in the following path:\n", save_path)
    #==========================================

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)    
    
    best_acc = 0
    
    for epoch in range(args.epochs):
 
        
        if epoch % args.validate_every == 0:
            acc = test(datasets["training"], datasets["validation"], model, accuracy_calculator, epoch=epoch, batch_size=args.batch_size)

            if acc[-1] > best_acc:
                if acc == 1.:
                    best_acc = 0.8 + epoch/10
                else:
                    best_acc = acc[-1]

                torch.save(model.state_dict(), f'./weights/pretrain_{args.model_name}_constrastive_freeze_and_layer_triple_miner.pth')
                
        pretrain(model=model,
                 train_loader=loaders["training"],
                 optimizer = optimizer,
                 epoch=epoch)
        
        

      # Testing
    test(datasets["training"], datasets["validation"], model, accuracy_calculator, epoch=epoch)
    test(datasets["training"], datasets["test"], model, accuracy_calculator, epoch=epoch)

    exit()
