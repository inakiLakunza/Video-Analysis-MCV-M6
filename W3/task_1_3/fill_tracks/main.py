import numpy as np
import pickle

from utils import make_video

if __name__ == "__main__":
    
    PICKLE_NAME = "./pickles/full_max_skip_20_det_th_07.pkl"
    with open(PICKLE_NAME, 'rb') as inp:
        ended_updater = pickle.load(inp)
    
    ORIGINAL_CSV_PATH = "/ghome/group07/test/W3/task_1_3/results_and_gt/task_1_3_RAFT_det_th_07_20_max_frame_skip.csv"
    
    DET_TH = "07"
    MAX_HOLE = 10
    UPDATED_IMGS_PATH = "./updated_frames_max_"+str(MAX_HOLE)+"_"+DET_TH+"/"
    OUT_CSV_PATH = "./predictions_max_hole_"+str(MAX_HOLE)+"_"+DET_TH+".csv"
    ORIGINAL_IMG_PATH = "/ghome/group07/test/W3/task_1_3/out_imgs__det_th_"+DET_TH+"_max_frame_skip_20"
    ended_updater.fill_missing_tracks(ORIGINAL_CSV_PATH, OUT_CSV_PATH, ORIGINAL_IMG_PATH, max_hole=MAX_HOLE, updated_imgs_path=UPDATED_IMGS_PATH)
    
    #make_video(img_folder="/ghome/group07/test/W3/task_1_3/fill_tracks/updated_frames_max_"+str(MAX_HOLE), name="filled_800_1200", start=800, end=1200)
    #make_video(img_folder="/ghome/group07/test/W3/task_1_3/fill_tracks/updated_frames_max_"+str(MAX_HOLE), name="filled_full")

