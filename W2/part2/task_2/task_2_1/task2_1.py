import sys
sys.path.append('../')
import os
import json
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from track_classes_and_functions import Detection, Tracks_2_1, nms

import numpy as np
from tqdm import tqdm
import copy

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


import torch

from pathlib import Path
import random 


if __name__ == "__main__":


    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
    }

    NAME_TO_CLASS = {
        "person": 0,
        "bicycle": 1,
        "car": 2
    }

    N_FRAMES = 2141
    FRAME_SET_PATH = "./../../../frame_dataset"
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

    for i in tqdm(range(N_FRAMES)):

        img_path = os.path.join(COLOR_FRAME_SET_PATH, str(i)+".png")
        img = cv2.imread(img_path)

        preds = predictor(img)

        # Keep only car predictions
        keep_cars_mask = preds["instances"].pred_classes == NAME_TO_CLASS["car"]
        bboxes, scores = preds["instances"].pred_boxes[keep_cars_mask], preds["instances"].scores[keep_cars_mask]
        n_wanted_classes = sum(keep_cars_mask)

        frame_detections = []
        for i_det in range(n_wanted_classes):
            det = Detection(i_det, bboxes[i_det], scores[i_det])
            frame_detections.append(det)

        #detections_after_nms = nms(frame_detections, detection_threshold, min_iou)

        track_updater.update_tracks(frame_detections, i)
        frame_tracks = track_updater.get_tracks()
        print(f"Frame {i} has a total number of {len(frame_tracks)} shown\n\n")

        for frame_track in frame_tracks:
            if frame_track.get_last_frame_id() == i:
                detection = frame_track.get_last_detection()
                bb_color = frame_track.get_color()
                bb = detection.get_bb()

                x_min, y_min, x_max, y_max = bb
                mins = int(x_min), int(y_min)
                maxs = int(x_max), int(y_max)
                img = cv2.rectangle(img, mins, maxs, bb_color, bb_thickness)
                
                # Draw a smaller rectangle for ID label
                id_label = f"ID: {frame_track.get_track_id()}"
                label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_width, label_height = label_size


                # Place the label at the top-left corner inside the bounding box
                label_position = (x_min, y_min - 10)
                label_bg_end = (int(x_min) + int(label_width) + 20, int(y_min) - int(label_height) - 20)
                img = cv2.rectangle(img, (int(x_min), int(y_min) - 5), label_bg_end, bb_color, -1)  # -1 for filled rectangle
                img = cv2.putText(img, id_label, (int(x_min) + 10, int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                

                
        out_path = os.path.join(out_img_path, "frame_"+str(i)+".png")
        cv2.imwrite(out_path, img)
    
    




            










    


