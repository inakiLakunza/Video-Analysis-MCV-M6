import os
import random

import cv2
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer

import xml.etree.ElementTree as ET
import utils
import json

from typing import *



def is_parked(element) -> bool:
    attribute = element.find("attribute")
    if attribute is None:
        return False
    elif attribute.text == "false":
        return False
    return True
    


class Dataset():

    def __init__(self,  frames_folder: str) -> None:
        self._frames_folder = frames_folder



    

    def map_xml2dict(self, xml_annotations_path, ignore_parked: bool = False, ignore_classes: bool = False) -> Dict:

        ## Thanks for the parser dear pau torras 
        ### https://github.com/mcv-m6-video/mcv-m6-2022-team1/blob/main/w2/data.py


        dataset = ET.parse(str(xml_annotations_path)).getroot()

        # Build coco-compliant dataset in JSON format
        if ignore_classes:
            labels = {
                "moving": 1,
            }
            last_label = 1
        else:
            labels = {}
            last_label = -1

        # FIXME: Hardcoded, but necessary to ensure all images appear on the gt
        frames = set([ii for ii in range(1, 2142)])
        ann_id = 0

        # Create the annotations field
        annotations = []
        for track in dataset.findall("track"):
            if ignore_classes:
                obj_label = 1
            else:
                if track.attrib["label"] not in labels:
                    last_label += 1
                    labels[track.attrib["label"]] = last_label
                obj_label = labels[track.attrib["label"]]

            for num, box in enumerate(track.findall("box")):
                if ignore_parked and track.attrib["label"] == "car":
                    continue

                # Keep track of images with annotations
                frame = int(box.attrib["frame"]) + 1
                frames.add(frame)

                # Generate a bounding box
                bbox = [
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]) - float(box.attrib["xtl"]),
                    float(box.attrib["ybr"]) - float(box.attrib["ytl"]),
                ]

                annotations.append({
                    "id": ann_id,
                    "image_id": frame,
                    "category_id": obj_label,
                    "bbox": bbox,
                    "segmentation": [],
                    "keypoints": [],
                    "num_keypoints": 0,
                    "score": 1,
                    "area": bbox[-2] * bbox[-1],
                    "iscrowd": 0
                })
                ann_id += 1

        # Create the images field
        images = []
        for ii in frames:
            images.append({
                "id": ii,
                "license": 1,
                "file_name": f"frames/{ii:05}.jpg",
                "height": 1080,
                "width": 1920,
                "date_captured": None,
            })

        # Create the categories field
        categories = []
        for name, cat_id in labels.items():
            categories.append({
                "id": cat_id,
                "name": name,
                "supercategory": "vehicle",
                "keypoints": [],
                "skeleton": [],
            })
        licenses = {
            "id": 1,
            "name": "Unknown",
            "url": "Unknown",
        }
        info = {
            "year": 2022,
            "version": "0.0",
            "description": "Hopefully I did not screw it up this time",
            "contributor": "Nobody",
            "url": "None",
        }

        coco_dict = {
            "info": info,
            "images": images,
            "categories": categories,
            "annotations": annotations,
            "licenses": licenses
        }

        return coco_dict
    

    def load_jsons(self, coco_dictionary):

        #### first 25% training #####
        start_validation_idx = int(len(coco_dictionary['images']) * .25)
        train, val = list(), list()
        using = train
        everything = []

        for idx, image in enumerate(coco_dictionary['images']):

            gt = [{**x, 'bbox_mode': 1} for x in coco_dictionary['annotations'] if image['id'] == x['image_id']]
  
            if idx > start_validation_idx: using = val
            using.append({**image, 'image_id': image['id'], 'annotations': gt})
            everything.append({**image, 'image_id': image['id'], 'annotations': gt})
            #if idx == 5:
            #    print(using)
            #    exit()

        open('datafolds/train_first.json', 'w').write(json.dumps(train))
        open('datafolds/everything.json', 'w').write(json.dumps(everything))
        open('datafolds/val_first.json', 'w').write(json.dumps(val))


        train, val = list(), list()
        using = train

        for idx, image in enumerate(coco_dictionary['images']):

            gt = [{**x, 'bbox_mode': 1} for x in coco_dictionary['annotations'] if image['id'] == x['image_id']]
            
            if random.random() > .25: using = val
            else: using = train
            using.append({**image, 'image_id': image['id'], 'annotations': gt})
        
        random.shuffle(train)
        random.shuffle(val)
        open('datafolds/train_random.json', 'w').write(json.dumps(train))
        open('datafolds/val_random.json', 'w').write(json.dumps(val))

        ### CROSS VALIDATION ####
        fold_1, fold_2, fold_3 = list(), list(), list()
        start_fold_2 = int(0.33 * len(coco_dictionary['images']))
        start_fold_3 = int(0.66 * len(coco_dictionary['images']))
        using = fold_1

        for idx, image in enumerate(coco_dictionary['images']):

            gt = [{**x, 'bbox_mode': 1} for x in coco_dictionary['annotations'] if image['id'] == x['image_id']]
            
            if idx > start_fold_3: using = fold_3
            elif idx > start_fold_2: using = fold_2
            else: using = fold_1

            using.append({**image, 'image_id': image['id'], 'annotations': gt})

        open('datafolds/val_1.json', 'w').write(json.dumps(fold_1))
        open('datafolds/val_2.json', 'w').write(json.dumps(fold_2))
        open('datafolds/val_3.json', 'w').write(json.dumps(fold_3))

        open('datafolds/train_13.json', 'w').write(json.dumps(fold_1 + fold_3))
        open('datafolds/train_12.json', 'w').write(json.dumps(fold_1 + fold_2))
        open('datafolds/train_32.json', 'w').write(json.dumps(fold_3 + fold_2))
        



if __name__ == '__main__':
    #unvideo_video('/home/adria/Desktop/mcv-m6-2023-team2/data/AICity_S03_c010/vdo.avi')
    #a = (generate_gt_from_xml('../data/AICity_S03_c010/ai_challenge_s03_c010-full_annotation.xml'))

    d = Dataset(frames_folder="../frame_dataset/color")
    annotations = d.map_xml2dict(xml_annotations_path='../../ai_challenge_s03_c010-full_annotation.xml', ignore_parked=False)
    d.load_jsons(annotations)
    print(annotations['categories'])