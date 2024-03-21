
import sys
import os

import json
import numpy as np
from pathlib import Path

import glob

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def map_labels_to_integers(labels, start):
    labels = np.array(labels)
    labels_copy = np.copy(labels)
    unique_labels = sorted(set(labels_copy))
    label_to_integer = {label: i+start for i, label in enumerate(unique_labels)}
    mapped_labels = [label_to_integer[label] for label in labels]
    return mapped_labels, unique_labels


def map_labels_to_integers2(labels, start):
    labels_copy = labels.copy()
    unique_labels = []
    label_to_integer = {}
    count = start
    for label in labels_copy:
        if label not in label_to_integer:
            label_to_integer[label] = count
            unique_labels.append(label)
            count += 1

    mapped_labels = [label_to_integer[label] for label in labels]
    return mapped_labels, unique_labels


def get_number_of_imgs_in_folder(path: Path, format: str = "png") -> int:
    png_files = glob.glob(os.path.join(path, "*."+format))
    return len(png_files)

def initialize_predictor(model_name: str = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
                         detection_threshold: int = 0.5) -> DefaultPredictor:
    
    cfg = get_cfg()
    
    # FASTER RCNN
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    # SET THRESHOLD FOR SCORING
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    return predictor


def check_file_exists(file_path: Path) -> bool:
    return os.path.isfile(file_path)

def check_folder_exists(folder_path: Path) -> bool:
    return os.path.isdir(folder_path)

def create_folder_if_not_exist(folder_path: Path) -> None:
    if not check_folder_exists(folder_path):
        print("The out img directory does not exist, creating a new one")
        os.makedirs(folder_path)
    else: 
        print("WARNING:\n the out img directory exists, it will be overwritten")


def delete_csv_if_exists(csv_path: Path) -> None:
    
    if check_file_exists(csv_path):
        print(f"""Deleting the following csv path
           because we will create a new one will be created:\n
          {csv_path}\n""")
        os.remove(csv_path)
    
