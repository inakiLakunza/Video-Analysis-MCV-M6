
import os
import sys

sys.path.append("../")

from pathlib import Path

import uuid
import csv
import cv2
import pickle
from time import time

from utils import *




if __name__ == "__main__":

    start_time = time()

    GT_PATH_ROOT = "/ghome/group07/test/W3/task_2/train"
    FRAMES_PATH_ROOT = "/ghome/group07/test/W4/frame_dataset_PNG"

    SEQUENCE = "1"

    cams_path = os.path.join(GT_PATH_ROOT, "S0"+SEQUENCE)
    videos: list[Path] = [os.path.join(cams_path, camera, "gt", "gt.txt") for camera in os.listdir(cams_path) if camera != ".DS_Store"]

    CROP_PATH_ROOT = "/ghome/group07/test/W4/part2/triplet_train/saved_crops"
    crop_save_path = os.path.join(CROP_PATH_ROOT, "S0"+SEQUENCE)
    
    PICKLE_PATH_ROOT = "/ghome/group07/test/W4/part2/triplet_train/pickles"

    save_dict = {}

    for video_path in videos:

        cam_name = video_path.split("/")[-3]
        cam_number = int(cam_name[2:])
        if cam_number > 28: 
            continue


        frames_path = os.path.join(FRAMES_PATH_ROOT, "S0"+SEQUENCE, cam_name, "color")
        n_frames = get_number_of_imgs_in_folder(frames_path)


        with open(video_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

        for i in range(len(rows)):
            frame_id = int(rows[i][0])-1
            track_id = rows[i][1]
            x_min, y_min = int(rows[i][2]), int(rows[i][3])
            width, height = int(rows[i][4]), int(rows[i][5])

            x_max = x_min+width
            y_max = y_min+height

            img_path = os.path.join(frames_path, str(frame_id)+".png")
            full_img = cv2.imread(img_path)
            print(img_path)
            crop = full_img[y_min:y_max, x_min:x_max, :]
            img_name = "S0"+SEQUENCE+"_"+cam_name+"_"+str(frame_id)+"_"+str(track_id)+".png"
            img_save_path = os.path.join(crop_save_path, img_name)
            cv2.imwrite(img_save_path, crop)
            print(f"Image saved in {img_save_path}")

            if track_id in save_dict:
                inner_list = save_dict[track_id].copy()
                inner_list.append(img_name)
                save_dict[track_id] = inner_list

            else:
                save_dict[track_id] = [img_name]



    pickle_path = os.path.join(PICKLE_PATH_ROOT, "S0"+SEQUENCE, "info.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"""\n\nPickle for sequence {str(SEQUENCE)} saved correctly,\n
           the needed time was: {time()-start_time}""")    
    
            

            

            


        




