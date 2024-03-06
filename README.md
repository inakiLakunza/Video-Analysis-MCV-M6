# C6: Video Analysis - Group 6

* Cristian Gutiérrez
* Iñaki Lacunza
* Carlos Boned
* Marco Cordón

## How to Run
Install dependencies via a `requirements.txt` file.

```
git clone https://github.com/mcv-m6-video/mcv-c6-2024-team6.git
cd mcv-c6-2024-team6/
python3 -m pip install -r requirements.txt
```

Please, download the data from UAB virtual campus `AICity_data` and move it to the current repo.

```
mv /path/to/your/AICity_data/ .
cd WX/taskX_X/
python3 main.py
```

## Week 1: Background estimation
This first week was focused on background estimation to be able to segment the moving objects. Thorough this lab we will work with the AICityData dataset.
- Task 1.1: Fixed Gaussian Estimation
- Task 1.2: Evaluation mAP
- Task 2.1: Adaptative Modelling
- Task 2.2: Comparison between fixed and adaptative
- Task 3: Comparison with SOTA models (CNT, LSBP, GMG, MOG, ...)
- Task 4: Color sequences

#### Some example results
Fixed Gaussian Modeling with an alpha of 2 for the first 60 frames of the sequence.

<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W1/task1/gifs/init_alpha_2.gif" width="600" height="auto">

## Week 2: Object detection and Tracking
This second week we had to implement and evaluate different SOTA models for object detection and tracking algorithms.
Our annotated sequence can be found at [/W2/part1/task_1_2/annotations.xml](https://raw.githubusercontent.com/mcv-m6-video/mcv-c6-2024-team6/main/W2/part1/task_1_2/annotations.xml).
- Task 1: Object Detection
    - Task 1.1: Off-the-shelf
    - Task 1.2: Annotation
    - Task 1.3: Fine-tune to our annotated sequence
    - Task 1.4: K-Fold Cross-validation
- Task 2: Object tracking
    - Task 2.1: Overlapping method
    - Task 2.2: Kalman filtering
    - Task 2.3: TrackEval Metrics
- Task 3: **(OPTIONAL)** CVPR 2021 AI City Challenge

#### Some example results

##### Tracking by Overlap:
<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W2/part2/task_2/task_2_1/gifs/video_task_2_1_det_th_05_1.gif" width="400" height="auto">

##### Tracking with Kalman Filtering:
<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W2/part2/task_2/task_2_2/gif_for_README/gif_README.gif" width="400" height="auto">

##### CVPR 2021 AI City Challenge
<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W2/part2/task_3/results_and_gt/example_gif.gif" width="400" height="auto">
