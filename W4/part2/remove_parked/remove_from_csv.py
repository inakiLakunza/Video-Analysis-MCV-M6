
import numpy as np
import pandas as pd
import csv





if __name__ == "__main__":

    csv_path = "/ghome/group07/test/W3/task_2/results_and_gt/task_2_3_10_RAFT_det_th_07_iou_01.csv"
    OUT_CSV_PATH = "./prueba.csv"

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    seen_tracks = {}

    for i in range(len(rows)):
        frame_id = rows[i][0]
        track_id = rows[i][1]

        x_min, y_min = float(rows[i][2]), float(rows[i][3])
        width, height = float(rows[i][4]), float(rows[i][5])

        centroid = (x_min+width/2, y_min+height/2)

        if track_id not in seen_tracks:
            seen_tracks[track_id] = {
                "first_frame": frame_id,
                "first_centroid": centroid,
                "last_frame": None,
                "last_centroid": None
            }

        else:
            inner_dict = seen_tracks[track_id].copy()
            inner_dict["last_frame"] = frame_id
            inner_dict["last_centroid"] = centroid
            seen_tracks[track_id] = inner_dict

    print(seen_tracks)

    tracks_to_maintain = []
    for track_id, track_info in seen_tracks.items():
        centroid_f = track_info["first_centroid"]
        centroid_l = track_info["last_centroid"]

        if not centroid_l: continue 

        if abs(centroid_l[0]-centroid_f[0])+abs(centroid_l[1]-centroid_f[1]) > 200:
            tracks_to_maintain.append(track_id)

    
    with open(OUT_CSV_PATH, "a") as write_file:
        writer = csv.writer(write_file)
        for i in range(len(rows)):

            track_id = rows[i][1]
            if track_id in tracks_to_maintain:

                

                writer.writerow(rows[i])
    