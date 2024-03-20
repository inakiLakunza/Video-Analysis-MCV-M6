import sys
import os
import json
from utils import *
import numpy as np
from tqdm import tqdm
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path
import detectron2
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
from sort import Sort
import raft

os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_mps(y_min, a=-0.0045, b=0.4):
    return b * np.exp(a * y_min)

def moving_avg(data, window_size=5):
    if len(data) < window_size:
        return sum(data) / len(data)
    else:
        window = data[-window_size:]
        return sum(window) / len(window)

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
    with open('../configs/configs_task2_2.json') as config:
        task_configs = json.load(config)

    # Params
    detection_threshold = 0.5
    min_iou = 0.5
    max_frames_skip = 15
    bb_thickness = 5
    out_img_path = "./frames"
    FPS = 10
    meters_per_pixel = 0.05
    previous_centroids = {}

    model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    res = []
    id_colors = {}
    id_motion = {}
    history_frames = {}
    velocities = {}

    tracker = Sort(max_age=1, min_hits=3)
    for i in tqdm(range(N_FRAMES)):
        img_path = os.path.join(COLOR_FRAME_SET_PATH, str(i)+".png")
        img = cv2.imread(img_path)
        preds = predictor(img)

        # Keep only car predictions
        keep_cars_mask = preds["instances"].pred_classes == NAME_TO_CLASS["car"]
        bboxes, scores = preds["instances"].pred_boxes[keep_cars_mask].tensor.cpu().numpy(), preds["instances"].scores[keep_cars_mask].cpu().numpy()
        n_wanted_classes = sum(keep_cars_mask)

        # SORT expects detections to be - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score]
        detections = np.hstack([bboxes, scores[:, None]])
        tracked = tracker.update(detections) # Returns the a similar array, where the last column is the object ID.

        print(f"Frame {i} has a total number of {len(tracked)} shown\n\n")
        
        for object in tracked:
            bbox = object[:4]
            identifier = int(object[-1])

            if identifier not in id_colors.keys():
                color = tuple(np.random.choice(range(256), size=3))
                id_colors[identifier] = tuple(map(int, color))

            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])

            if y_min < 100:
                continue

            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if identifier not in id_motion.keys():
                id_motion[identifier] = [centroid]
            else:
                id_motion[identifier].append(centroid)
            
            if identifier in previous_centroids:
                prev_cx, prev_cy = previous_centroids[identifier]
                displacement = np.sqrt((centroid[0] - prev_cx)**2 + (centroid[1] - prev_cy)**2)
                velocity = displacement * 10 # pixel/seconds
                meters_per_pixel = compute_mps(y_min)
                real_velocity = velocity * meters_per_pixel * 3.6 # km/h
                if identifier in velocities:
                    velocities[identifier].append(real_velocity)
                else:
                    velocities[identifier] = [real_velocity]
                real_velocity = moving_avg(velocities[identifier], window_size=10)
                if real_velocity < 10:
                    continue
                id_label = f"ID: {identifier} ({real_velocity:.4f} km/h)"
            else:
                id_label = f"ID: {identifier} (Stationary)"

            previous_centroids[identifier] = centroid

            # Draw a smaller rectangle for ID label
            label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_width, label_height = label_size

            #print(f"({x_min}, {y_min}), ({x_max}, {y_max}), {id_colors[identifier]}")
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), id_colors[identifier], 5)

            for i in range(1, len(id_motion[identifier])):
                pt1 = (int(id_motion[identifier][i - 1][0]), int(id_motion[identifier][i - 1][1]))
                pt2 = (int(id_motion[identifier][i][0]), int(id_motion[identifier][i][1]))
                cv2.line(img, pt1, pt2, id_colors[identifier], 5)

            label_position = (x_min, y_min - 10)
            label_bg_end = (int(x_min) + int(label_width) + 20, int(y_min) - int(label_height) - 20)
            cv2.rectangle(img, (int(x_min), int(y_min) - 5), label_bg_end, id_colors[identifier], -1)  # -1 for filled rectangle
            cv2.putText(img, id_label, (int(x_min) + 10, int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        res.append(img)
        out_path = os.path.join(out_img_path, "frame_"+str(i)+".png")
        cv2.imwrite(out_path, img)
    make_video(np.array(res))







            










    


