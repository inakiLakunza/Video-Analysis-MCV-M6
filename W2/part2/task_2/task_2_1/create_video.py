import numpy as np
from utils import make_video


if __name__ == "__main__":
    
    out_img_path1 = "/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1/det_th_05"
    make_video(img_folder=out_img_path1, start=800, end=950, name="video_task_2_1_det_th_05_1", out_folder="/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1")
    print("video1 done")
    make_video(img_folder=out_img_path1, start=450, end=600, name="video_task_2_1_det_th_05_2", out_folder="/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1")
    print("video2 done")
    make_video(img_folder=out_img_path1, start=1550, end=1700, name="video_task_2_1_det_th_05_3", out_folder="/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1")
    print("video3 done")

    out_img_path2 = "/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1/det_th_07"
    make_video(img_folder=out_img_path2, start=800, end=950, name="video_task_2_1_det_th_07_1", out_folder="/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1")
    print("video4 done")
    make_video(img_folder=out_img_path2, start=450, end=600, name="video_task_2_1_det_th_07_2", out_folder="/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1")
    print("video5 done")
    make_video(img_folder=out_img_path2, start=1550, end=1700, name="video_task_2_1_det_th_07_3", out_folder="/ghome/group07/test/W2/part2/task_2_1/outs_with_labels_2_1")
    print("video6 done")
