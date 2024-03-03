import numpy as np

from pycocotools.mask import toBbox


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

    

class Detection():
    def __init__(self, frame_id, bb):
        self.frame_id = frame_id

        self.bb = bb.tensor.cpu().numpy()[0]
        self.box_x_min = self.bb[0]
        self.box_y_min = self.bb[1]
        self.box_x_max = self.bb[2]
        self.box_y_max = self.bb[3]

    def get_bb(self):
        return self.bb

    def get_box_values(self):
        return self.box_x_min, self.box_y_min, self.box_x_max, self.box_y_max
    
    def compute_iou(self, other):
        # Calculating coordinates of the intersection rectangle
        intersection_x1 = max(self.box_x_min, other.box_y_min)
        intersection_y1 = max(self.box_y_min, other.box_y_min)
        intersection_x2 = min(self.box_x_max, other.box_x_max)
        intersection_y2 = min(self.box_y_max, other.box_y_max)

        # Calculating area of intersection rectangle
        intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)

        # Calculating areas of each bounding box
        area_box1 = (self.box_x_max-self.box_x_min) * (self.box_y_max-self.box_y_min)
        area_box2 = (other.box_x_max-other.box_x_min) * (other.box_y_max-other.box_y_min)

        # Calculating union area
        union_area = area_box1 + area_box2 - intersection_area

        # Calculating IoU
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

        new_tracks = []
        for detection in new_detections:
            best_iou, best_iou_index, best_detection = -1, None, None
            for i in range(self.n_active_tracks):
                last_dect, last_frame = self.active_tracks[i].get_last_detection_and_frame_id()

                iou = last_dect.compute_iou(detection)
                if iou >= self.min_iou and iou>best_iou:
                    best_iou = iou
                    best_iou_index = i
                    best_detection = detection

            # IF THE TRACK ALREADY EXISTS UPDATE IT
            if best_iou_index:
                track = self.active_tracks[best_iou_index]
                track.add_detection_and_frame_id(best_detection, frame_id)
            
            # ELSE CREATE NEW TRACK
            else:
                new_t = self.new_track(frame_id, detection)
                new_tracks.append(new_t)
        
        
        
        self.active_tracks.extend(new_tracks)
        self.n_active_tracks = len(self.active_tracks)

                        

                    

            

        
            



        
        
        


        




