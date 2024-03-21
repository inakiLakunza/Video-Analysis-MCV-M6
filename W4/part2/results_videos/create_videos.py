import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import random
import os
import sys
from tqdm import tqdm

from random import randint


def read_video(folder_path: str, gt):
    files = []
    for f in os.listdir(folder_path):
        if f.endswith('.png'):
            files.append(f)
    # sortea por el nombre del png (frame)
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    #print(files)
    #sys.exit()

    frames = []
    color_frames = []
    for file in tqdm(files):
        # print(file.split('.'))
        # print(gt.keys())
        # print(gt[(file.split('.')[0])])
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        if img is not None:
            # Guardamos gray y color por si acaso (evitable)
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Pintar el gt
            frame_num = int(file.split('.')[0])
            if frame_num in gt.keys():
                # print(frame_num)
                bboxes = gt[frame_num]
                for bbox in bboxes:
                    # print(f'bbox: {bbox}')
                    xtl = int(bbox['xtl'])
                    ytl = int(bbox['ytl'])
                    xbr = int(bbox['xbr'])
                    ybr = int(bbox['ybr'])
                    track_id = int(bbox['track_id'])

                    if track_id in map_id_color:
                        track_color = map_id_color[track_id]
                    else:
                        r = randint(0, 255)
                        g = randint(0, 255)
                        b = randint(0, 255)
                        track_color = (r, g, b)
                        map_id_color[track_id] = track_color

                    frame_rgb = cv2.rectangle(frame_rgb, (xtl, ytl), (xbr, ybr), track_color, 4)

                    # Draw a smaller rectangle for ID label
                    id_label = f"ID: {track_id}"
                    label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_width, label_height = label_size


                    # Place the label at the top-left corner inside the bounding box
                    label_bg_end = (int(xtl) + int(label_width) + 20, int(ytl) - int(label_height) - 20)
                    frame_rgb = cv2.rectangle(frame_rgb, (int(xtl), int(ytl) - 5), label_bg_end, track_color, -1)  # -1 for filled rectangle
                    frame_rgb = cv2.putText(frame_rgb, id_label, (int(xtl) + 10, int(ytl) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            color_frames.append(frame_rgb)
            frames.append(frame_gray)

    return np.array(frames), np.array(color_frames)


def read_annotations(annotations_path: str):
    """
    frame_id, track_id, xtl, ytl, width, height, -1, -1, -1, -1
    """
    car_boxes = {}

    with open(annotations_path, 'r') as file:
        for line in file:
            elements = line.strip().split(',')
            if len(elements) >= 6:
                frame, track_id, xtl, ytl, w, h = elements[:6]
                box_attributes = {
                    "xtl": float(xtl),
                    "ytl": float(ytl),
                    "xbr": float(xtl) + float(w),
                    "ybr": float(ytl) + float(h),
                    "track_id": track_id
                }
                # Convert frame to int for consistent indexing
                frame = int(frame)
                if frame in car_boxes:
                    car_boxes[frame].append(box_attributes)
                else:
                    car_boxes[frame] = [box_attributes]

    return car_boxes


def make_video(estimation, video_name, folder_name):
    """
    Make a .mp4 from the estimation
    https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

    Parameters
        estimation : np.ndarray([1606, 1080, 1920, 3], dtype=uint8)
    """
    size = estimation.shape[1], estimation.shape[2]
    duration = estimation.shape[0]
    fps = 10
    out = cv2.VideoWriter(f'./{folder_name}/{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for i in range(duration):
        data = (estimation[i]).astype(np.uint8)
        # I am converting the data to gray but we should look into this...
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        out.write(data)
    out.release()
    print("Video done.")



if __name__ == '__main__':


    map_id_color = {}

    WANTED_TH = 6
    cams_to_analyze = ["c014", "c010"]

    ROOT_CSV_PATH = os.path.join("/ghome/group07/test/W4/part2/out_csvs", "th_"+str(WANTED_TH))


    for cam in cams_to_analyze:

        csv_path = os.path.join(ROOT_CSV_PATH, cam+".csv")
        csv_file= read_annotations(csv_path)

        images_path = '/ghome/group07/test/W3/task_2/frame_dataset/S03/'+cam+'/color/'
        _, color_frames = read_video(images_path, csv_file)
        folder_name = f'out_videos_th{WANTED_TH}'
        os.makedirs(f'./{folder_name}/', exist_ok=True)
        make_video(color_frames, "video_"+cam+"_3", folder_name)
    
