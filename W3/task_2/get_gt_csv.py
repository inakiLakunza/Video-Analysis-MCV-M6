import sys
sys.path.append('../')
import os

os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
import cv2
import datetime
from lxml import etree



def read_annotations(annotations_path: str):
    """
    Function to read the GT annotations from ai_challenge_s03_c010-full_annotation.xml

    At the moment we will only check that the track is for "car" and has "parked" as false
    and we will save the bounding box attributes from the 'box' element.
    """
    tree = etree.parse(annotations_path)
    root = tree.getroot()
    #car_boxes = {}

    result_file_path = "./results_and_gt/gt.csv"
    # IF CSV FILE EXISTS, DELETE IT:
    try:
        os.remove(result_file_path)
    except OSError:
        pass

    with open(result_file_path, "a") as file:
        for track in root.xpath(".//track[@label='car']"):
            track_id = track.get("id")
            for box in track.xpath(".//box"):
                frame = box.get("frame")
                box_attributes = {
                    "xtl": box.get("xtl"),
                    "ytl": box.get("ytl"),
                    "xbr": box.get("xbr"),
                    "ybr": box.get("ybr"),
                    # in the future we will need more attributes
                }
                w = float(box_attributes["xbr"])-float(box_attributes["xtl"])
                h = float(box_attributes["ybr"])-float(box_attributes["ytl"])
                #if frame in car_boxes:
                #    car_boxes[frame].append(box_attributes)
                #else:
                #    car_boxes[frame] = [box_attributes]

                line = f'{int(frame)+1}, {int(track_id)+1}, {float(box_attributes["xtl"])+1.}, {float(box_attributes["ytl"])+1.}, {w}, {h}, -1, -1, -1, -1\n'
                
                file.write(line)

    
if __name__ == "__main__":
    annotations_path = "/ghome/group07/test/W3/task_2/annotations.xml"
    read_annotations(annotations_path)