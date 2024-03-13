import numpy as np
import pickle

from utils import make_video

if __name__ == "__main__":
    
    with open('full_norm_and_moved.pkl', 'rb') as inp:
        ended_updater = pickle.load(inp)
    
    ORIGINAL_CSV_PATH = "/ghome/group07/test/W3/task_1_3/results_and_gt/task_1_3_RAFT_box_corrected.csv"
    
    MAX_HOLE = 20
    UPDATED_IMGS_PATH = "./updated_frames_max_"+str(MAX_HOLE)+"/"
    OUT_CSV_PATH = "./predictions_max_hole_"+str(MAX_HOLE)+".csv"
    ended_updater.fill_missing_tracks(ORIGINAL_CSV_PATH, OUT_CSV_PATH, max_hole=MAX_HOLE, updated_imgs_path=UPDATED_IMGS_PATH)
    
    make_video(img_folder="/ghome/group07/test/W3/task_1_3/fill_tracks/updated_frames_max_"+str(MAX_HOLE), name="filled_800_1200", start=800, end=1200)
    make_video(img_folder="/ghome/group07/test/W3/task_1_3/fill_tracks/updated_frames_max_"+str(MAX_HOLE), name="filled_full")

