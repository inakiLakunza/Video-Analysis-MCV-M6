import sys
sys.path.append('../')
import os
import json
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from track_classes_and_functions import *

import numpy as np
from tqdm import tqdm
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path
import cv2
import copy


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
import uuid

from time import time





def print_and_get_resume() -> dict:
    # Print resume
    print("Resume:")
    print(f"TRY_NAME: {TRY_NAME}")
    print(f"FINE_TUNED: {FINE_TUNED}")
    print(f"DETECTION_THRESHOLD: {DETECTION_THRESHOLD}")
    print(f"MIN_IOU: {MIN_IOU}")
    print(f"MAX_FRAMES_SKIP: {MAX_FRAMES_SKIP}")
    print(f"BB_THICKNESS: {BB_THICKNESS}")
    print(f"SEQUENCE_ROOT_PATH: {SEQUENCE_ROOT_PATH}")
    print(f"SEQUENCE_NUMBER: {SEQUENCE_NUMBER}")
    #print(f"OUT_CSV_ROOT_PATH: {OUT_CSV_ROOT_PATH}")
    print(f"WRITE_CSV: {WRITE_CSV}")
    print(f"WRITE_PKL: {WRITE_PKL}")
    print(f"WRITE_IMGS: {WRITE_IMGS}")
    #print(f"OUT_IMG_ROOT_PATH: {OUT_IMG_ROOT_PATH}")
    #print(f"OUT_IMG_PATH: {OUT_IMG_FOLDER}")
    #print(f"OUT_PICKLE_ROOT_PATH: {OUT_PICKLE_ROOT_PATH}")
    #print(f"OUT_PICKLE_PATH: {OUT_PICKLE_PATH}")
    

    # Create and return dictionary
    resume_dict = {
        "TRY_NAME": TRY_NAME,
        "FINE_TUNED": FINE_TUNED,
        "DETECTION_THRESHOLD": DETECTION_THRESHOLD,
        "MIN_IOU": MIN_IOU,
        "MAX_FRAMES_SKIP": MAX_FRAMES_SKIP,
        "BB_THICKNESS": BB_THICKNESS,
        "SEQUENCE_ROOT_PATH": SEQUENCE_ROOT_PATH,
        "SEQUENCE_NUMBER": SEQUENCE_NUMBER,
        #"OUT_CSV_ROOT_PATH": OUT_CSV_ROOT_PATH,
        "WRITE_CSV": WRITE_CSV,
        "WRITE_PKL": WRITE_PKL,
        "WRITE_IMGS": WRITE_IMGS
        #"OUT_IMG_ROOT_PATH": OUT_IMG_ROOT_PATH,
        #"OUT_IMG_FOLDER": OUT_IMG_FOLDER,
        #"OUT_PICKLE_ROOT_PATH": OUT_PICKLE_ROOT_PATH,
        #"OUT_PICKLE_PATH": OUT_PICKLE_PATH,
        
    }
    return resume_dict




def track_update_loop(track_updater: Track_Updater, predictor: DefaultPredictor, n_frames: int, img_path: Path) -> Track_Updater:
    
    for i in tqdm(range(n_frames)):

        i_path = os.path.join(img_path, str(i)+".png")
        print(i_path)
        img = cv2.imread(i_path)
        img = copy.deepcopy(img)
        preds = predictor(img)

        # Keep only car predictions
        car_mask = preds["instances"].pred_classes == NAME_TO_CLASS["car"]
        truck_mask = preds["instances"].pred_classes == NAME_TO_CLASS["truck"]
        bus_mask = preds["instances"].pred_classes == NAME_TO_CLASS["bus"]
        keep_wanted_classes_mask = torch.any(torch.stack((car_mask, truck_mask, bus_mask), dim=0), dim=0)
        bboxes, scores = preds["instances"].pred_boxes[keep_wanted_classes_mask], preds["instances"].scores[keep_wanted_classes_mask]
        n_wanted_classes = sum(keep_wanted_classes_mask)


        frame_detections = []
        for i_det in range(n_wanted_classes):
            det = Detection(i_det, bboxes[i_det], scores[i_det])
            frame_detections.append(det)

        # Non Maximum Supression
        frame_detections, frame_scores = nms_otaku(frame_detections, 0.2)
        nms_detections = []
        for i_det in range(len(frame_detections)):
            det = Detection(i_det, frame_detections[i_det], frame_scores[i_det], tensor=False)
            nms_detections.append(det)


        track_updater.update_tracks(nms_detections, i)
        frame_tracks = track_updater.get_tracks()
        print(f"Frame {i} has a total number of {len(frame_tracks)} shown\n\n")



        if WRITE_CSV:
            with open(OUT_CSV_PATH, "a") as file:
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


        if WRITE_IMGS:
            for frame_track in frame_tracks:
                if frame_track.get_last_frame_id() == i:
                    detection = frame_track.get_last_detection()
                    bb_color_normal = frame_track.get_color()
                    bb = detection.get_bb()

                    x_min, y_min, x_max, y_max = bb
                    mins = int(x_min), int(y_min)
                    maxs = int(x_max), int(y_max)
                    img = cv2.rectangle(img, mins, maxs, bb_color_normal, BB_THICKNESS)
                    
                    # Draw a smaller rectangle for ID label
                    id_label = f"ID: {frame_track.get_track_id()}"
                    label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_width, label_height = label_size


                    # Place the label at the top-left corner inside the bounding box
                    label_bg_end = (int(x_min) + int(label_width) + 20, int(y_min) - int(label_height) - 20)
                    img = cv2.rectangle(img, (int(x_min), int(y_min) - 5), label_bg_end, bb_color_normal, -1)  # -1 for filled rectangle
                    img = cv2.putText(img, id_label, (int(x_min) + 10, int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out_path = os.path.join(OUT_IMG_FOLDER, "frame_"+str(i)+".png")
            cv2.imwrite(out_path, img)


    return track_updater



def analyze_video(path: Path) -> Track_Updater:

    print(path)
    n_frames = get_number_of_imgs_in_folder(path)
    print(f"n_frames to analyze: {n_frames}")

    predictor = initialize_predictor(detection_threshold=DETECTION_THRESHOLD)


    track_updater = Track_Updater(MIN_IOU, MAX_FRAMES_SKIP)
    track_updater = track_update_loop(track_updater=track_updater, predictor=predictor,
                                      n_frames=n_frames, img_path=path)




if __name__ == "__main__":


    CLASS_NAMES: dict[int, str] = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        5: 'bus',
        7: 'truck'
    }

    NAME_TO_CLASS: dict[str, int]= {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'bus': 5,
        'truck': 7
    }


    with open('./../../configs/single_camera_track_conf.json') as config:
        configs = json.load(config)

    TRY_NAME: str = configs["try_name"]
    if TRY_NAME == "":
        TRY_NAME = uuid.uuid4()

    FINE_TUNED: bool = configs["fine_tuned"]
    
    DETECTION_THRESHOLD: int = configs["detection_threshold"]
    

    MIN_IOU: int = configs["min_iou"]
    MAX_FRAMES_SKIP: int = configs["max_frames_skip"]

    BB_THICKNESS: int = configs["bb_thickness"]
    

    SEQUENCE_ROOT_PATH: str = configs["sequence_root_path"]
    SEQUENCE_NUMBER: int = configs["sequence_number"]
    error_msg_seq_number: str = f"The chosen sequence number is not correct, it must be 1, 3 or 4, and you inserted {SEQUENCE_NUMBER}"
    assert SEQUENCE_NUMBER in [1, 3, 4], error_msg_seq_number

    WRITE_PKL: bool = configs["write_pkl"]
    WRITE_CSV: bool = configs["write_csv"]
    WRITE_IMGS: bool = configs["write_imgs"]


    resume_dict: dict = print_and_get_resume()

    videos_path: Path = os.path.join(SEQUENCE_ROOT_PATH, "S0"+str(SEQUENCE_NUMBER) )
    videos: list[Path] = [os.path.join(videos_path, camera, "color") for camera in os.listdir(videos_path)]

    cam_times = {}
    for i_vid in range(len(videos)):
        video_path = videos[i_vid]

        cam_name = video_path.split("/")[0]

        
        if WRITE_CSV: 
            OUT_CSV_ROOT_PATH: str = configs["out_csv_root_path"]
            OUT_CSV_PATH: Path = os.path.join(OUT_CSV_ROOT_PATH, str(SEQUENCE_NUMBER), cam_name)
            create_folder_if_not_exist(OUT_CSV_PATH)
            OUT_CSV_PATH: Path = os.path.join(OUT_CSV_PATH, TRY_NAME+".csv")
            delete_csv_if_exists(OUT_CSV_PATH)

        
        if WRITE_IMGS:
            OUT_IMG_ROOT_PATH: str = configs["out_img_root_path"]
            OUT_IMG_FOLDER: Path = os.path.join(OUT_IMG_ROOT_PATH, "S0"+str(SEQUENCE_NUMBER), cam_name, TRY_NAME)
            create_folder_if_not_exist(OUT_IMG_FOLDER)


        start_time = time()
        print(f"Analyzing video number {i_vid}, which is the following :\n {video_path}\n")
        track_updater = analyze_video(video_path)
        end_time = time() - start_time
        print(f"Camera {cam_name} analyzed, needed {end_time} seconds")
        if WRITE_CSV: print(f"csv file saved in {OUT_CSV_PATH}")
        if WRITE_IMGS: print(f"out imgs saved in {OUT_IMG_FOLDER}")
        cam_times[cam_name] = end_time

        if WRITE_PKL:
            OUT_PICKLE_ROOT_PATH: str = configs["out_pickle_root_path"]
            OUT_PICKLE_PATH: Path = os.path.join(OUT_PICKLE_ROOT_PATH, str(SEQUENCE_NUMBER), TRY_NAME+".pkl")
            if check_file_exists(OUT_PICKLE_PATH):
                print(f"""The following pickle exists: {OUT_PICKLE_ROOT_PATH},
                    so it will be rewritten""")
                with open('file.pkl', 'wb') as file: 
                    pickle.dump(track_updater, OUT_PICKLE_PATH) 
    
    resume_dict["cam_times"] = cam_times
    
    SAVE_RESULTS_PATH = configs["save_results_info_path"]
    SAVE_RESULTS_PATH = os.path.join(SAVE_RESULTS_PATH, "results_info_S0"+str(SEQUENCE_NUMBER)+".json")
    data = json.load(open(SAVE_RESULTS_PATH))
    # convert data to list if not
    if type(data) is dict:
        data = [data]

    # append new item to data lit
    data.append(resume_dict)

    # write list to file
    with open(SAVE_RESULTS_PATH, 'w') as outfile:
        json.dump(data, outfile, indent = 4)

    sys.exit()


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
    out_img_path = task_configs["out_img_path"]

    # FASTER RCNN
    model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    # SET THRESHOLD FOR SCORING
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    result_file_path = "/ghome/group07/test/W3/task_2/results_and_gt/task_2_4_28_RAFT_det_th_07_iou_005.csv"
    # IF CSV FILE EXISTS, DELETE IT:
    try:
        os.remove(result_file_path)
    except OSError:
        pass



    track_updater = Tracks_2_1(min_iou, max_frames_skip)
    id_motion = {}
    #for i in tqdm(range(N_FRAMES)):
    for i in tqdm(range(N_FRAMES)):

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