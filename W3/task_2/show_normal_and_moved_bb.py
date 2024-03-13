import sys
sys.path.append('../')
import os
import json
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from track_classes_and_functions import Detection, Tracks_2_1, nms_otaku

import numpy as np
from tqdm import tqdm
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path
import copy

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
import cv2
import torch

from pathlib import Path
import random 


if __name__ == "__main__":


    CLASS_NAMES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        7: 'truck'
    }

    NAME_TO_CLASS = {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'truck': 7
    }

    N_FRAMES = 2414
    FRAME_SET_PATH = "/ghome/group07/test/W3/task_2/frame_dataset/S03/c013"
    COLOR_FRAME_SET_PATH = os.path.join(FRAME_SET_PATH, "color")
    GRAY_FRAME_SET_PATH = os.path.join(FRAME_SET_PATH, "gray")


    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

    # MASK RCNN
    #model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    with open('/ghome/group07/test/W3/configs/configs_task2_1.json') as config:
        task_configs = json.load(config)
    detection_threshold = task_configs["detection_threshold"]
    min_iou = task_configs["min_iou"]
    max_frames_skip = task_configs["max_frames_skip"]
    bb_thickness = task_configs["bb_thickness"]
    #out_img_path = task_configs["out_img_path"]
    out_img_path = "/ghome/group07/test/W3/task_2/bbs_normal_and_rect"

    # FASTER RCNN
    model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    # SET THRESHOLD FOR SCORING
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    # EXAMPLE
    #----------------------------------------------------------------------
    #try_img_path = "/ghome/group07/test/W2/frame_dataset/color/0.png"
    #try_img = cv2.imread(try_img_path)
    #output = predictor(try_img)

    #print("Ouput of pred_classes:\n", output["instances"].pred_classes)
    #print("\n\n\nOutput of instances: \n", output["instances"].pred_boxes)
    #print("\n\n\nWhole output:\n", output)

    #save_img(try_img, output, "predicted_img_example.png", cfg)
    #----------------------------------------------------------------------

    track_updater = Tracks_2_1(min_iou, max_frames_skip)
    id_motion = {}
    #for i in tqdm(range(N_FRAMES)):
    for i in tqdm(range(750, 800)):

        img_path = os.path.join(COLOR_FRAME_SET_PATH, str(i)+".png")
        img = cv2.imread(img_path)
        img_copy = copy.deepcopy(img)
        preds = predictor(img)

        # Keep only car predictions
        car_mask = preds["instances"].pred_classes == NAME_TO_CLASS["car"]
        truck_mask = preds["instances"].pred_classes == NAME_TO_CLASS["truck"]
        keep_cars_and_trucks_mask = torch.any(torch.stack((car_mask, truck_mask), dim=0), dim=0)
        #keep_cars_and_trucks_mask = ((preds["instances"].pred_classes == NAME_TO_CLASS["car"]) or (preds["instances"].pred_classes == NAME_TO_CLASS["truck"]))
        bboxes, scores = preds["instances"].pred_boxes[keep_cars_and_trucks_mask], preds["instances"].scores[keep_cars_and_trucks_mask]
        n_wanted_classes = sum(keep_cars_and_trucks_mask)

        # MAYBE WE SHOULD REMOVE SOME BB USING A THRESHOLD,
        # BUT I THINK THAT THIS IS DONE IN LINE 82 ALREADY USING THAT THRESHOLD
        # OTHERWISE WE SHOULD JUST APPLY A THRESHOLD

        frame_detections = []
        for i_det in range(n_wanted_classes):
            det = Detection(i_det, bboxes[i_det], scores[i_det])
            frame_detections.append(det)

        frame_detections, frame_scores = nms_otaku(frame_detections, 0.2)
        nms_detections = []
        for i_det in range(len(frame_detections)):
            det = Detection(i_det, frame_detections[i_det], frame_scores[i_det], tensor=False)
            nms_detections.append(det)


        track_updater.update_tracks(nms_detections, i)
        frame_tracks = track_updater.get_tracks()
        print(f"Frame {i} has a total number of {len(frame_tracks)} shown\n\n")

        for frame_track in frame_tracks:
            if frame_track.get_last_frame_id() == i:
                detection = frame_track.get_last_detection()
                bb_color_normal = (0,0,255)
                bb = detection.get_bb()

                x_min, y_min, x_max, y_max = bb
                mins = int(x_min), int(y_min)
                maxs = int(x_max), int(y_max)
                img = cv2.rectangle(img, mins, maxs, bb_color_normal, bb_thickness)
                
                # Draw a smaller rectangle for ID label
                id_label = f"ID: {frame_track.get_track_id()}"
                label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_width, label_height = label_size


                # Place the label at the top-left corner inside the bounding box
                label_position = (x_min, y_min - 10)
                label_bg_end = (int(x_min) + int(label_width) + 20, int(y_min) - int(label_height) - 20)
                img = cv2.rectangle(img, (int(x_min), int(y_min) - 5), label_bg_end, bb_color_normal, -1)  # -1 for filled rectangle
                img = cv2.putText(img, id_label, (int(x_min) + 10, int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                bb_color_displaced = (255, 0, 0)
                displacement_x, displacement_y = detection.of_direction
                print(detection.of_direction)
                x_min, y_min, x_max, y_max = bb
                x_min += displacement_x 
                y_min += displacement_y
                x_max += displacement_x
                y_max += displacement_y
                mins = int(x_min), int(y_min)
                maxs = int(x_max), int(y_max)
                img = cv2.rectangle(img, mins, maxs, bb_color_displaced, bb_thickness)
                
                # Draw a smaller rectangle for ID label
                id_label = f"ID: {frame_track.get_track_id()}"
                label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_width, label_height = label_size


                # Place the label at the top-left corner inside the bounding box
                label_position = (x_min, y_min - 10)
                label_bg_end = (int(x_min) + int(label_width) + 20, int(y_min) - int(label_height) - 20)
                img = cv2.rectangle(img, (int(x_min), int(y_min) - 5), label_bg_end, bb_color_displaced, -1)  # -1 for filled rectangle
                img = cv2.putText(img, id_label, (int(x_min) + 10, int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



                
        out_path = os.path.join(out_img_path, "frame_"+str(i)+".png")
        cv2.imwrite(out_path, img)
    



            










    


