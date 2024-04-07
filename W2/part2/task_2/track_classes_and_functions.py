import numpy as np

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
    def __init__(self, frame_id, bb, score):
        self.frame_id = frame_id

        self.bb = bb.tensor.cpu().numpy()[0]
        self.box_x_min = self.bb[0]
        self.box_y_min = self.bb[1]
        self.box_x_max = self.bb[2]
        self.box_y_max = self.bb[3]

        self.score = score.cpu().numpy()

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
    


    def update_tracks(self, new_detections, frame_id):

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


        for detection in new_detections:
            best_iou, best_iou_index, best_detection = -1, -1, None
            #print("n_active tracks: ", self.n_active_tracks)
            for i in range(self.n_active_tracks):
                last_dect, last_frame = self.active_tracks[i].get_last_detection_and_frame_id()

                iou = last_dect.compute_iou(detection)
                #print("iou s:       ", iou, best_iou)
                if iou >= self.min_iou and iou>best_iou:
                    best_iou = iou
                    best_iou_index = i
                    best_detection = detection

            # IF THE TRACK ALREADY EXISTS UPDATE IT
            #print("best iou index:      ", best_iou_index)
            if best_iou_index > -1:
                track = self.active_tracks[best_iou_index]
                track.add_detection_and_frame_id(best_detection, frame_id)
            
            # ELSE CREATE NEW TRACK
            else:
                self.new_track(frame_id, detection)
        
                
        self.n_active_tracks = len(self.active_tracks)
        print(f"Number of active tracks: f{self.n_active_tracks}  , frame number: {frame_id}")

                        
# NON-MAXIMUM SUPRESSION
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

    return detection_list_thresholded
                            



            

        
            



        
        
        


        




