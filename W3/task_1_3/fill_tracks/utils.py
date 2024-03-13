
import numpy as np
from collections import Counter
import cv2
import datetime
from lxml import etree
import matplotlib.pyplot as plt
import random
import os



# HABRÁ QUE PONER TMB MEDIAN Y MEAN
def flow_votation(box_flow):
    #print(box_flow)
    # Reshape the flow array to 2D (height*width, 2) for counting
    flow_reshaped = box_flow.reshape(-1, 2)
    #print(flow_reshaped)

    # Count the occurrences of each unique flow vector
    flow_counts = Counter(map(tuple, flow_reshaped))

    #print(flow_counts)

    # Get the most frequent flow vector
    most_common_flow = flow_counts.most_common(1)[0][0]

    return most_common_flow


def flow_median(box_flow):

    flow_reshaped = box_flow.reshape(-1, 2)
    median_x = np.median(flow_reshaped[:,1])
    median_y = np.median(flow_reshaped[:,0])

    return (median_y, median_x)


def flow_mean(box_flow):

    flow_reshaped = box_flow.reshape(-1, 2)
    mean_x = np.mean(flow_reshaped, axis=1)
    mean_y = np.mean(flow_reshaped, axis=0)

    return (mean_y, mean_x)


def read_video(vid_path: str):
    vid = cv2.VideoCapture(vid_path)
    frames = []
    color_frames = []
    while True:
        ret, frame = vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            color_frames.append(frame_rgb)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        else:
            break
    vid.release()
    return np.array(frames), np.array(color_frames)


def read_annotations(annotations_path: str, parked: bool):
    """
    Function to read the GT annotations from ai_challenge_s03_c010-full_annotation.xml

    At the moment we will only check that the track is for "car" and has "parked" as false
    and we will save the bounding box attributes from the 'box' element.
    """
    tree = etree.parse(annotations_path)
    root = tree.getroot()
    car_boxes = {}

    for track in root.xpath(".//track[@label='car']"):
        track_id = track.get("id")
        for box in track.xpath(".//box"):
            parked_attribute = box.find(".//attribute[@name='parked']")
            if parked_attribute is not None and parked_attribute.text == str(parked).lower():
                frame = box.get("frame")
                box_attributes = {
                    "xtl": box.get("xtl"),
                    "ytl": box.get("ytl"),
                    "xbr": box.get("xbr"),
                    "ybr": box.get("ybr"),
                    # in the future we will need more attributes
                }
                if frame in car_boxes:
                    car_boxes[frame].append(box_attributes)
                else:
                    car_boxes[frame] = [box_attributes]

    return car_boxes


def split_frames(frames):
    """
    Returns 25% and 75% split partition of frames.
    """
    split = int(frames.shape[0] * 0.25)
    return frames[:split], frames[split:]


def make_video(out_folder="./", img_folder=None, name="example_video", start=0, end=2140):
    """
    Make a .mp4 from the estimation
    https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

    Parameters
        estimation : np.ndarray([1606, 1080, 1920, 3], dtype=uint8)
    """

    if img_folder is None:
        img_folder = out_folder
    
    duration = 2141
    img_shape = 1920, 1080 
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video=cv2.VideoWriter(os.path.join(out_folder, name+".avi"), fourcc, fps, (img_shape[0], img_shape[1]), True)

    
    for j in range(start, end):
        img = cv2.imread(os.path.join(img_folder, "frame_"+str(j)+".png"))
        video.write(img)