import torch
import sys
import numpy as np
import json

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset, WrapperDataloader
from torchvision import models
import os
from sklearn.metrics import average_precision_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from metrics import mapk
#from losses import ContrastiveLoss, TripletLoss
import pytorch_metric_learning
from pytorch_metric_learning import losses, trainers
from pytorch_metric_learning import samplers
import cv2 as cv

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer





class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(512, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x.to(self.device)
        x = x.squeeze(-1).squeeze(-1)
        x = self.activation(x)
        x = self.linear(x)
        return x


class Faster_embedding(torch.nn.Module):
    def __init__(self, embed_size=2096, num_epochs=50, batch_size= 32, loss= losses.TripletMarginLoss(), FT_DATASET_NAME ="COCO") :
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initializing Model with {}\n".format(self.device))


        # Model
        self.cfg = get_cfg()
        model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

        self.cfg.defrost()
        self.cfg.merge_from_file(model_zoo.get_config_file(model))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

        embed = EmbeddingLayer(embed_size)
        self.cfg = torch.nn.Sequential(*list(self.cfg.children())[:-1], embed)

        self.cfg.to(self.device)
        self.save_path = './Faster_weights/'

        self.cfg.DATASETS.TRAIN = (FT_DATASET_NAME + "train2014",)
        self.cfg.DATASETS.TEST = (FT_DATASET_NAME + "val2014",)

        self.cfg.INPUT.MASK_FORMAT = "bitmask"
        self.cfg.DATALOADER.NUM_WORKERS = 4

        output_dir = "/ghome/group07/C5-W3/faster_output"
        self.cfg.OUTPUT_DIR = output_dir

        with open('./configs/task_e_train_config.json') as train_info:
            self.info = json.load(train_info)


    def extract_features(self, dataloader):
        features = []
        targets = []
        for batch_idx, (images, labels, captions) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.forward(images)
            features.append(outputs.cpu().detach().numpy())
            targets.append(labels.cpu().numpy())
        return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)
    

    def train_model(self,lr=0.002, batch_size_per_image=32, imgs_per_batch=10):
        self.cfg.SOLVER.BASE_LR = lr
        self.cfg.SOLVER.MAX_ITER = self.info["max_iter"]
        self.cfg.SOLVER.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        self.cfg.SOLVER.IMS_PER_BATCH = imgs_per_batch
        self.cfg.SOLVER.CHECKPOINT_PERIOD = self.info["checkpoint_period"]
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.8
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()


    def train_knn(self, dataloader):
        features, labels = self.extract_features(dataloader)
        clf = KNeighborsClassifier(n_neighbors=13,n_jobs=-1,metric='euclidean')
        clf.fit(features, labels)
        return clf , labels
    

    def test(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_0001499.pth")  # path to the model we just trained
        predictor = DefaultPredictor(self.cfg)


        # EVALUATE THE MODEL
        evaluator = COCOEvaluator(
                    self.FT_DATASET_NAME + "val",
                    output_dir=str(self.output_dir),
        )

        val_loader = build_detection_test_loader(self.cfg, self.FT_DATASET_NAME + "val")


        print(inference_on_dataset(predictor.model, val_loader, evaluator))
