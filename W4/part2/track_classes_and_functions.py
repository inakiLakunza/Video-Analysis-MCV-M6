import os
import sys
sys.path.insert(0, '/ghome/group07/test/W3/task_1_2')

from utils import *
from box_flow_compute_methods import *

import numpy as np
import time
from RAFT import main as raft_query



def compute_raft(im1_path: str, im2_path: str):
    channels = 3
    kwargs = {}
    s = time.time()
    #print("Entering raft")
    flow = raft_query.compute_raft_pipeline(im1_path, im2_path, channels, **kwargs)
    #print(f"Flow is {flow.shape}")
    e = time.time()
    return flow, (e - s)




class Track():
    def __init__(self, track_updater: 'Track_Updater', first_detection: 'Detection', first_frame_id: int):
        self.id: int = track_updater.n_total_tracks
        self.color: tuple[int] = list(np.random.choice(range(256), size=3))
        self.first_detection: 'Detection' = first_detection
        self.detections: list['Detection'] = [first_detection]
        self.frames: list[int] = [first_frame_id]

        self.of_directions: list[tuple[float]] = [first_detection.get]

    def add_detection_and_frame_id(self, detection: 'Detection', frame_id: int) -> None:
        self.detections.append(detection)
        self.frames.append(frame_id)
        self.of_directions.append(detection.get_of_direction())

    def get_last_detection_and_frame_id(self) -> tuple['Detection', int]:
        return self.detections[-1], self.frames[-1]

    def get_last_detection(self) -> 'Detection':
        return self.detections[-1]
    
    def get_last_frame_id(self) -> int:
        return self.frames[-1]
    
    def get_detections(self) -> list['Detection']:
        return self.detections
    
    def get_frames(self) -> list[int]:
        return self.frames

    def get_color(self) -> tuple[int]:
        return (int(self.color[0]), int(self.color[1]), int(self.color[2]))

    def get_track_id(self) -> int:
        return self.id

    def get_of_directions(self) -> list[tuple[float]]:
        return self.of_directions




    

class Detection():
    def __init__(self, frame_id: int, bb: list[float], score: float, tensor: bool = True):
        self.frame_id = frame_id

        if tensor:
            self.bb = bb.tensor.cpu().numpy()[0]
            self.score = score.cpu().numpy()
        else:
            self.bb = bb
            self.score = score

        self.box_x_min = self.bb[0]
        self.box_y_min = self.bb[1]
        self.box_x_max = self.bb[2]
        self.box_y_max = self.bb[3]

        centroid_x = (self.box_x_min + self.box_x_max) / 2
        centroid_y = (self.box_y_min + self.box_y_max) / 2

        self.centroid = (centroid_x, centroid_y)

        # WE INITIALIZE IT WITH 0, AND COMPUTE IT LATER
        self.of_direction = None


    def get_bb(self) -> list[int]:
        return self.bb

    def get_box_values(self) -> tuple[float]:
        return self.box_x_min, self.box_y_min, self.box_x_max, self.box_y_max
    
    def get_score(self) -> float:
        return self.score
    
    def get_centroid(self) -> tuple[float]:
        return self.centroid
    
    def get_of_direction(self) -> tuple[float]:
        return self.of_direction

    def compute_iou(self, other) -> float:
         # Get the coordinates of the intersection rectangle
        x_left = max(self.box_x_min, other.box_x_min)
        y_top = max(self.box_y_min, other.box_y_min)
        x_right = min(self.box_x_max, other.box_x_max)
        y_bottom = min(self.box_y_max, other.box_y_max)

        # Calculate the area of intersection rectangle
        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        # Calculate the area of both squares
        square1_area = (self.box_x_max - self.box_x_min + 1) * (self.box_y_max - self.box_y_min + 1)
        square2_area = (other.box_x_max - other.box_x_min + 1) * (other.box_y_max - other.box_y_min + 1)

        # Calculate the Union area
        union_area = square1_area + square2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou
    

    def compute_moved_iou(self, detection_to_move, frame_id) -> float:

        scale = frame_id-detection_to_move.frame_id
        if scale >= 5:
            scale = 5

        movement_x = detection_to_move.of_direction[0]*scale
        movement_y = detection_to_move.of_direction[1]*scale

        # Get the coordinates of the intersection rectangle
        x_left = max(self.box_x_min, detection_to_move.box_x_min+movement_x)
        y_top = max(self.box_y_min, detection_to_move.box_y_min+movement_y)
        x_right = min(self.box_x_max, detection_to_move.box_x_max+movement_x)
        y_bottom = min(self.box_y_max, detection_to_move.box_y_max+movement_y)

        # Calculate the area of intersection rectangle
        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        # Calculate the area of both squares
        square1_area = (self.box_x_max - self.box_x_min + 1) * (self.box_y_max - self.box_y_min + 1)
        square2_area = (detection_to_move.box_x_max - detection_to_move.box_x_min + 1) * (detection_to_move.box_y_max - detection_to_move.box_y_min + 1)

        # Calculate the Union area
        union_area = square1_area + square2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou
    


class Movement_Forward():

    def __init__(self, track: Track, frame_id: int, detection: Detection):
        self.track: Track = track
        self.detection: Detection = detection
        self.color: tuple[int] = track.get_color()
        self.frame_id: int = frame_id

        self.track_frame_ids = track.get_frames()
        self.track_of_list = track.get_of_directions()

    def get_track(self) -> Track:
        return self.track
    
    def get_detection(self) -> Detection:
        return self.detection
    
    def get_color(self) -> tuple[int]:
        return self.color
    
    def get_frame_id(self) -> int:
        return self.frame_id

    
    def compute_movement_forward(self) -> 'Movement_Forward':
        
        reversed_frame_ids = self.track_frame_ids.copy()
        reversed_frame_ids.reverse()

        reversed_track_of_list = self.track_of_list.copy()
        reversed_track_of_list.reverse()

        
        










class Track_Updater():
    def __init__(self, min_iou, max_no_detect):
        self.min_iou = min_iou
        self.max_not_detected_tracks_to_end = max_no_detect
        self.n_total_tracks = 0
        self.n_active_tracks = 0
        self.n_ended_tracks = 0

        self.total_tracks = []
        self.active_tracks = []
        self.ended_tracks = []


    def new_track(self, frame_id, detection):
        new_Track = Track(self, detection, frame_id)

        self.n_total_tracks +=1
        self.n_active_tracks +=1
        
        self.total_tracks.append(new_Track)
        self.active_tracks.append(new_Track)

        return new_Track

    def get_tracks(self):
        return self.active_tracks
    


    def compute_OF_present_future(self, frame_id):

        if frame_id == 0:
            return np.zeros((1080, 1920, 2), np.float32)

        FRAME_FOLDER = "/ghome/group07/test/W2/frame_dataset/color"
        PAST_IMG_ID = os.path.join(FRAME_FOLDER, str(frame_id-1)+".png")
        PRESENT_IMG_ID = os.path.join(FRAME_FOLDER, str(frame_id)+".png")
        
        flow, _ = compute_raft(PAST_IMG_ID, PRESENT_IMG_ID)

        return flow


    def assign_of_direction(self, flow, detection, method="median"):

        method_to_use = method.lower()

        assert method.lower() in ["median", "mean", "votation"]

        x_min = detection.box_x_min 
        y_min = detection.box_y_min 
        x_max = detection.box_x_max 
        y_max = detection.box_y_max


        flow_within_bbox = flow[int(y_min):int(y_max), int(x_min):int(x_max), :]

        
        if method_to_use == "median": bb_direction = flow_median(flow_within_bbox)
        elif method_to_use == "mean": bb_direction = flow_mean(flow_within_bbox)
        elif method_to_use == "votation": bb_direction = flow_votation(flow_within_bbox)

        detection.of_direction = bb_direction
            

        '''    
            x_min_disp = x_min + bb_direction[0]
            x_max_disp = y_min + bb_direction[0]
            y_min_disp = x_min + bb_direction[1]
            y_max_disp = y_min + bb_direction[1]
            disp_bb = [x_min_disp, y_min_disp, x_max_disp, y_max_disp]

            disp_det = Detection(detection.frame_id, disp_bb, detection.score)

            displaced_detections.append(disp_det)

        return displaced_detections
        '''



    def update_tracks(self, new_detections, frame_id, moved=True, method="median"):

        # OPTICAL FLOW WITH PREVIOUS FRAME
        flow = self.compute_OF_present_future(frame_id)
        #print(flow.shape)

        live_tracks = []
        for track in self.active_tracks:
            last_detection, last_frame_id = track.get_last_detection_and_frame_id()

            if frame_id-last_frame_id > self.max_not_detected_tracks_to_end:
                self.n_ended_tracks += 1
                self.ended_tracks.append(track)
            else:
                live_tracks.append(track)
                
        self.active_tracks = live_tracks
        self.n_active_tracks = len(live_tracks)



        #new_tracks = []
        for detection in new_detections:
            best_iou, best_iou_index, best_detection = -1, -1, None
            #print("n_active tracks: ", self.n_active_tracks)
            for i in range(self.n_active_tracks):
                last_dect, last_frame = self.active_tracks[i].get_last_detection_and_frame_id()

                if moved: iou = detection.compute_moved_iou(last_dect, frame_id)
                else: iou = last_dect.compute_iou(detection)

                #print("iou s:       ", iou, best_iou)
                if iou >= self.min_iou and iou>best_iou:
                    best_iou = iou
                    best_iou_index = i
                    best_detection = detection

            # IF THE TRACK ALREADY EXISTS UPDATE IT
            #print("best iou index:      ", best_iou_index)
            if best_iou_index > -1:

                self.assign_of_direction(flow, best_detection)

                track = self.active_tracks[best_iou_index]
                track.add_detection_and_frame_id(best_detection, frame_id)
            
            # ELSE CREATE NEW TRACK
            else:
                self.assign_of_direction(flow, detection)

                self.new_track(frame_id, detection)
        
        
                
        self.n_active_tracks = len(self.active_tracks)
        print(f"Number of active tracks: f{self.n_active_tracks}  , frame number: {frame_id}")

                        
# NON-MAXIMUM SUPRESSION
# (No MameS) XD  XDDDDDD
# TAKEN FROM (https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536)
def nms(detections, conf_threshold, iou_threshold):
    detection_list_thresholded = []
    detection_list_new = []

    # Stage1: Sort boxes and filter out boxes with low confidence
    detections_sorted  = sorted(detections, reverse=True, key = lambda x : x.score)
    for detection in detections_sorted:
        if detection.score > conf_threshold:
            detection_list_thresholded.append(detection)
    
    # Stage 2: Loop over all boxes, and remove boxes with high IOU
    while len(detection_list_thresholded) > 0:
        current_detection = detection_list_thresholded.pop(0)
        detection_list_new.append(current_detection)
        for detection in detection_list_thresholded:
            if current_detection.score == detection:
                iou = current_detection.compute_iou(detection)
                if iou > iou_threshold:
                    detection_list_thresholded.remove(detection)

    return detection_list_new


# TAKEN FROM:
# https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
def nms_otaku(detections, threshold):
    
    bounding_boxes = []
    confidence_score = []
    for detection in detections:
        bounding_boxes.append(detection.bb)
        confidence_score.append(detection.score)

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score
                            



            

        
            



        
        
        


        




