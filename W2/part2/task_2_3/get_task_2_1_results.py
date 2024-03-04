import sys
sys.path.append('../')
import os
import json
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from utils import *
from track_classes_and_functions import Detection, Tracks_2_1, nms

import numpy as np
from tqdm import tqdm
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path
import cv2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
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

from pycocotools.mask import toBbox

import torch

from pathlib import Path
import random 


if __name__ == "__main__":


    CLASS_NAMES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
    }

    NAME_TO_CLASS = {
        'person': 0,
        'bicycle': 1,
        'car': 2
    }

    N_FRAMES = 2141
    FRAME_SET_PATH = "/ghome/group07/test/W2/frame_dataset"
    COLOR_FRAME_SET_PATH = os.path.join(FRAME_SET_PATH, "color")
    GRAY_FRAME_SET_PATH = os.path.join(FRAME_SET_PATH, "gray")


    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

    # MASK RCNN
    #model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    with open('./../configs/configs_task2_1.json') as config:
        task_configs = json.load(config)
    detection_threshold = task_configs["detection_threshold"]
    min_iou = task_configs["min_iou"]
    max_frames_skip = task_configs["max_frames_skip"]
    bb_thickness = task_configs["bb_thickness"]
    out_img_path = task_configs["out_img_path"]

    # FASTER RCNN
    model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    # SET THRESHOLD FOR SCORING
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    result_file_path = "./results_and_gt_csv/task_2_1.csv"
    # IF CSV FILE EXISTS, DELETE IT:
    try:
        os.remove(result_file_path)
    except OSError:
        pass



    track_updater = Tracks_2_1(min_iou, max_frames_skip)
    for i in tqdm(range(N_FRAMES)):

        img_path = os.path.join(COLOR_FRAME_SET_PATH, str(i)+".png")
        img = cv2.imread(img_path)
        preds = predictor(img)

        # Keep only car predictions
        keep_cars_mask = preds["instances"].pred_classes == NAME_TO_CLASS["car"]
        bboxes, scores = preds["instances"].pred_boxes[keep_cars_mask], preds["instances"].scores[keep_cars_mask]
        n_wanted_classes = sum(keep_cars_mask)

        # MAYBE WE SHOULD REMOVE SOME BB USING A THRESHOLD,
        # BUT I THINK THAT THIS IS DONE IN LINE 82 ALREADY USING THAT THRESHOLD
        # OTHERWISE WE SHOULD JUST APPLY A THRESHOLD

        frame_detections = []
        for i_det in range(n_wanted_classes):
            det = Detection(i_det, bboxes[i_det], scores[i_det])
            frame_detections.append(det)

        #detections_after_nms = nms(frame_detections, detection_threshold, min_iou)

        track_updater.update_tracks(frame_detections, i)
        frame_tracks = track_updater.get_tracks()
        
        
        with open(result_file_path, "a") as file:
            for frame_track in frame_tracks:
                last_frame_id = frame_track.get_last_frame_id()
                if last_frame_id == i:
                    frame_id = frame_track.get_track_id()
                    detection = frame_track.get_last_detection()
                    score = detection.get_score()

                    bb = detection.get_bb()

                    x_min, y_min, x_max, y_max = bb
                    width = x_max-x_min
                    height = y_max-y_min

                    line = f"{i+1}, {frame_id+1}, {x_min+1}, {y_min+1}, {width}, {height}, {score}, -1, -1, -1\n"
                    file.write(line)









            










    


