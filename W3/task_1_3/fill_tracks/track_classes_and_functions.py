import os
import sys
sys.path.insert(0, '/ghome/group07/test/W3/task_1_2')

from utils import *

import numpy as np
import time
from RAFT import main as raft_query
from tqdm import tqdm
import pandas as pd
import csv

import copy



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
    def __init__(self, track_updater, first_detection, first_frame_id):
        self.id = track_updater.n_total_tracks
        self.color = list(np.random.choice(range(256), size=3))
        self.first_detection = first_detection
        self.detections = [first_detection]
        self.frames = [first_frame_id]

    def add_detection_and_frame_id(self, detection, frame_id):
        self.detections.append(detection)
        self.frames.append(frame_id)

    def get_last_detection_and_frame_id(self):
        return self.detections[-1], self.frames[-1]

    def get_last_detection(self):
        return self.detections[-1]
    
    def get_last_frame_id(self):
        return self.frames[-1]
    
    def get_detections(self):
        return self.detections
    
    def get_frames(self):
        return self.frames

    def get_color(self):
        return (int(self.color[0]), int(self.color[1]), int(self.color[2]))

    def get_track_id(self):
        return self.id

    

class Detection():
    def __init__(self, frame_id, bb, score, tensor=True):
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

        

        # WE INITIALIZE IT WITH 0, AND COMPUTE IT LATER
        self.of_direction = None


    def get_bb(self):
        return self.bb

    def get_box_values(self):
        return self.box_x_min, self.box_y_min, self.box_x_max, self.box_y_max
    
    def get_score(self):
        return self.score

    def compute_iou(self, other):
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
    

    def compute_moved_iou(self, detection_to_move, frame_id):

        scale = frame_id-detection_to_move.frame_id
        if scale >= 5:
            scale = 5

        movement_x = detection_to_move.of_direction[0]*scale
        movement_y = detection_to_move.of_direction[1]*scale

        print("movements (x,y): ", movement_x, movement_y)

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
    

class Tracks_2_1():
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


    def assign_of_direction(self, flow, detection):

        x_min = detection.box_x_min 
        y_min = detection.box_y_min 
        x_max = detection.box_x_max 
        y_max = detection.box_y_max

        #print(y_min, y_max, x_min, x_max)

        #print(flow.shape)
        #print(flow)
        flow_within_bbox = flow[int(y_min):int(y_max), int(x_min):int(x_max), :]
        #print(flow_within_bbox)
        # HABRÁ QUE PONER TMB MEDIAN Y MEAN
        bb_direction = flow_median(flow_within_bbox)
        detection.of_direction = bb_direction
                    


    def update_tracks(self, new_detections, frame_id):

        # OPTICAL FLOW WITH PREVIOUS FRAME
        flow = self.compute_OF_present_future(frame_id)

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

                #iou = last_dect.compute_iou(detection)
                iou = detection.compute_moved_iou(last_dect, frame_id)

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




    def add_missing_track_to_csv(self, csv_path, out_csv_path, frame_id, track_id, x_min, y_min, x_max, y_max, score, written=False):

        # REMEMBER THAT THE GT AND THE CSV IS DONE WITH BASE 1,
        # SO WE HAVE TO FIND THE NEXT FRAME NUMBER
        displaced_frame_id = frame_id+1

        if not written:
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
            
        else: 
            with open(out_csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)

        if os.path.exists(out_csv_path):
            os.remove(out_csv_path)

        width = x_max-x_min
        height = y_max-y_min

        for i in range(len(rows)):
             if int(rows[i][0]) == displaced_frame_id:
                 new_list = rows[:i]
                 
                 new_line = rows[i].copy()
                 new_line[1]=" "+str(int(track_id)+1)
                 new_line[2]=" "+str(float(x_min)+1)
                 new_line[3]=" "+str(float(y_min)+1)
                 new_line[4]=" "+str(width)
                 new_line[5]=" "+str(height)
                 new_line[6]=" "+str(score)

                 new_list.append(new_line)
                 new_list.extend(rows[i:])

                 print(f"\nInserted new line in csv file, the line is the following:\n")
                 print(new_line)
                 print("\n")
                 break
            
        with open(out_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(new_list)


    def scale_diff(self, i_diff):
        if i_diff < 5: return i_diff
        elif i_diff < 10: return 7.5
        elif i_diff < 15: return 10
        elif i_diff < 30: return 15
        elif i_diff < 60: return 20
        else: return 25

    def fill_missing_tracks(self, original_csv_path, out_csv_path, max_hole=10, updated_imgs_path="/ghome/group07/test/W3/task_1_3/fill_tracks/updated_frames"):

        img_folder = "/ghome/group07/test/W3/task_1_3/bbs_normal_and_rect"        
        updated_folder = updated_imgs_path
        bb_color = (0, 255, 0)

        updated_frames = []
        copied_frames = []

        written = False

        if len(self.active_tracks) > 0:
            self.ended_tracks.extend(self.active_tracks)


        for track in tqdm(self.ended_tracks):

            last_frame = track.frames[0]
            actual_frame = track.frames[0]

            i_last_frame = 0
            for i in range(len(track.frames)):
                actual_frame = track.frames[i]

                diff = actual_frame-last_frame
                if diff>1 and diff<max_hole:

                    track_id = track.id
                    last_detection = track.detections[i_last_frame]
                    last_det_of_dir = last_detection.of_direction
                    disp_x = last_det_of_dir[1]
                    disp_y = last_det_of_dir[0]
                    score = last_detection.get_score()

                    for i_diff in range(1, diff):
                        frame_to_mod_id = last_frame+i_diff
                        bb_to_move = last_detection.bb
                        x_min = bb_to_move[0]
                        y_min = bb_to_move[1]
                        x_max = bb_to_move[2]
                        y_max = bb_to_move[3]

                        

                        if abs(disp_x) < 0.005 and abs(disp_y) < 0.005:
                            x_min_disp = x_min
                            y_min_disp = y_min
                            x_max_disp = x_max
                            y_max_disp = y_max

                        else:
                            scaling = self.scale_diff(i_diff)
                            x_min_disp = x_min+disp_x*scaling
                            x_max_disp = x_max+disp_x*scaling
                            y_min_disp = y_min+disp_y*scaling
                            y_max_disp = y_max+disp_y*scaling


                        self.add_missing_track_to_csv(original_csv_path, out_csv_path, frame_to_mod_id, track_id, x_min, y_min, x_max, y_max, score, written=written)
                        written=True


                        if frame_to_mod_id in updated_frames:
                            loaded_img_path = os.path.join(updated_folder, "frame_"+str(frame_to_mod_id)+".png")
                        else:
                            loaded_img_path = os.path.join(img_folder, "frame_"+str(frame_to_mod_id)+".png")
                            print(loaded_img_path)
                        
                        img = cv2.imread(loaded_img_path)
                        img_copy = copy.deepcopy(img)

                        mins = int(x_min_disp), int(y_min_disp)
                        maxs = int(x_max_disp), int(y_max_disp)
                        img_copy = cv2.rectangle(img_copy, mins, maxs, bb_color, 5)
                        
                        # Draw a smaller rectangle for ID label
                        id_label = f"ID: {track_id}"
                        label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        label_width, label_height = label_size

                        # Place the label at the top-left corner inside the bounding box
                        #label_position = (x_min, y_min - 10)
                        label_bg_end = (int(x_min) + int(label_width) + 20, int(y_min) - int(label_height) - 20)
                        img_copy = cv2.rectangle(img_copy, (int(x_min), int(y_min) - 5), label_bg_end, bb_color, -1)  # -1 for filled rectangle
                        img_copy = cv2.putText(img_copy, id_label, (int(x_min) + 10, int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                        out_path = os.path.join(updated_folder, "frame_"+str(frame_to_mod_id)+".png")
                
                        cv2.imwrite(out_path, img_copy)
                        print(f"img {frame_to_mod_id} has been updated, added detection of track {track_id}")
                        updated_frames.append(frame_to_mod_id)

                else:
                    if actual_frame not in updated_frames and actual_frame not in copied_frames:
                        loaded_img_path = os.path.join(img_folder, "frame_"+str(actual_frame)+".png")
                        img = cv2.imread(loaded_img_path)
                        img_copy = copy.deepcopy(img)
                        out_path = os.path.join(updated_folder, "frame_"+str(actual_frame)+".png")
                        cv2.imwrite(out_path, img_copy)
                        copied_frames.append(actual_frame)


                last_frame = actual_frame
                i_last_frame = i

        print("FINISHED UPDATING TRACKS")

















# NON-MAXIMUM SUPRESSION
# (No MameS) XD  XDDDDDD
# TAKEN FROM (https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536)
def nms(detections, conf_threshold, iou_threshold):
    detection_list_thresholded = []
    detection_list_new = []

    # Stage1: Sort boxes and filter out boxes with low confidence
    detections_sorted  = sorted(detections, reverse=True, key = lambda x : x.score)
    for detection in detections_sorted:
        print(detection.score, conf_threshold)
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
                            



            

        
            



        
        
        


        




