import numpy as np

from pycocotools.mask import toBbox


class Track():
    def __init__(self, id, first_detection, first_frame_id):
        self.id = id
        self.color = list(np.random.choice(range(256), size=3))
        self.first_detection = first_detection
        self.detections = [first_detection]
        self.frames = [first_frame_id]

    def add_detection_and_frame(self, detection, frame_id):
        self.detections.append(detection)
        self.frames.append(frame_id)

    def get_last_detection_and_frame(self):
        return self.detections[-1], self.frames[-1]
    
    def get_detections(self):
        return self.detections
    
    def get_frames(self):
        return self.frames

    

class Detection():
    def __init__(self, track_id, frame_id, bb):
        self.track_id = track_id
        self.frame_id = frame_id

        self.bb = bb
        self.tlx = bb[0]
        self.tly = bb[1]
        self.brx = bb[2]
        self.bry = bb[3]

    def get_bb(self):
        return self.bb

    def get_tl(self):
        return self.tlx, self.tly
    def get_br(self):
        return self.brx, self.bry
    



class Tracks_2_1():
    def __init__(self, min_iou, max_no_detect):
        self.min_iou = min_iou
        self.max_no_detected_tracks_to_end = max_no_detect
        self.n_total_tracks = 0
        self.n_active_tracks = 0
        self.n_ended_tracks = 0

        self.total_tracks = []
        self.active_tracks = []
        self.n_ended_tracks = []

    def new_track(self, track_id, frame_id, bb):
        new_Detection = Detection(track_id, frame_id, bb)
        new_Track = Track(self.n_total_tracks, new_Detection, frame_id)

        self.n_total_tracks +=1
        self.n_active_tracks +=1
        
        self.total_tracks.append(new_Track)
        self.active_tracks.append(new_Track)

        # SHOULD WE RETURN THE DETECTION AND THE TRACK?

    def updated_tracks():
        # TO DO
        pass


        




