import pickle
import sys
import random
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize

import sys
import os

import json
from pathlib import Path
import tqdm

import glob

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_number_of_imgs_in_folder(path: Path, format: str = "png") -> int:
    png_files = glob.glob(os.path.join(path, "*."+format))
    return len(png_files)

def save_custom_data(chosen_set, path_ids, path_labels):
    imgs_ids = []
    label_dict = {}
    for object_label in chosen_set:
        if object_label==str(89): print(chosen_set[object_label])
        for img_id in chosen_set[object_label]:
            imgs_ids.append(img_id)
            if img_id not in label_dict.keys():
                label_dict[img_id] = [object_label]
            else:
                inside_list = label_dict[img_id]
                inside_list.append(object_label)
                label_dict[img_id] = inside_list


    imgs_ids=list(set(imgs_ids))

    with open(path_ids, "wb") as f:
        pickle.dump(imgs_ids, f)

    with open(path_labels, "wb") as f:
        pickle.dump(label_dict, f)


# UTILS POS-NEG ====================================================================

def search_pos_neg(dict, curr_img_id, curr_labels):
    """
    This function is used to search for positive and negatives samples.
    Suppose we have:
        187464: ['49', '78', '79', '50', '82', '81', '44']
    Then we need to find a sample that does not contain any of the above labels (neg image)
    And a sample that contains the most of them (pos image)
    """
    pos_img_id = None
    neg_img_id = None
    max_common = 0
    min_common = float('inf')
    for img_id, labels in sorted(dict.items(), key=lambda x: random.random()):
        # Positive sample
        common_labels = list(set(curr_labels).intersection(labels))
        if len(common_labels) > max_common and img_id != curr_img_id:
            pos_img_id = img_id
            max_common = len(common_labels)
            if max_common == len(curr_labels): break
    
    pos_labels = dict[pos_img_id]
    curr_labels = curr_labels + pos_labels
    for img_id, labels in sorted(dict.items(), key=lambda x: random.random()):
        # Negative sample
        common_labels = list(set(curr_labels).intersection(labels))
        if len(common_labels) < min_common:
            neg_img_id = img_id
            min_common = len(common_labels)
            if min_common == 0: break
    return pos_img_id, neg_img_id


def create_image_loader(dict):
    """
    Will return a dataloader with:
        (anchor_id, pos_id, neg_id, anchor labels)
    """
    res = []
    for img_id, labels in tqdm.tqdm(dict.items()):
        # search for positive and negative samples
        pos_id, neg_id = search_pos_neg(dict, img_id, labels)
        res.append((img_id, pos_id, neg_id, labels))
    return res


def plot_prec_rec_curve_multiclass(y_gt, y_pred, output, n_classes):
    y_gt_bin = label_binarize(y_gt, classes=np.arange(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(n_classes))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_gt_bin[:, i], y_pred_bin[:, i])
        average_precision[i] = average_precision_score(y_gt_bin[:, i], y_pred_bin[:, i])
        
    # Plot each class
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} AP={average_precision[i]:.2f}')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: mAP={np.mean(list(average_precision.values())):.2f}')
    plt.legend(loc='best')
    plt.savefig(output)
    plt.close()



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
