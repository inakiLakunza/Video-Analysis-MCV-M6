# C6: Video Analysis - Group 6

* Cristian Gutiérrez
* Iñaki Lacunza
* Carlos Boned
* Marco Cordón

Final slides: 

- Part 1: [Link to Google Slides part 1](https://docs.google.com/presentation/d/1R14EFnpQF_S54wvmZ-PL2Aki6zctgfgduNtRsQr8WNc/edit?usp=sharing)

- Part 2: [Link to Google Slides part 2](https://docs.google.com/presentation/d/1h8Kzy6xpP4Zjnm_cwTqMZAASYlCs1FcJ5b0cF8cXIhM/edit?usp=sharing)

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

##### Kalman filtering:
<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W2/part2/task_2/task_2_2/out_det_05.gif" width="400" height="auto">

##### CVPR 2021 AI City Challenge
<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W2/part2/task_3/results_and_gt/example_gif.gif" width="400" height="auto">

## Week 3: Optical Flow
On this week the main goal has been to estimate the optical flow of a video sequence and try to improve an object tracking algorithm using the optical flow.

- Task 1: Optical Flow.
    - Estimate the Optical Flow with block matching.
    - Estimate the Optical Flow with off-the-shelf method.
    - Improve the object tracking algorithm with Optical Flow.
- Task 2: Multi-Target Single-Camera tracking (MTSC).
    - Evaluate our best tracking algorithm in different SEQs of AI City Challenge.
    - Evaluate the tracking using IDF1 and HOTA scores.

#### Some example results

##### Task 1.3
<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W3/task_1_3/gifs_and_imgs/normal_and_rect_530_650.gif" width="400" height="auto">

##### Task 2
<img src="https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W3/task_2/gif_read_me.gif" width="400" height="auto">


## Week 4: Speed estimation and Multi-Camera Tracking
This final week consisted in two separate tasks, first one was to estimate the velocity, and the second one was to perform Multi-Camera Tracking (MCT).
- Task 1: Speed Estimation
    - Task 1.1: Rudimentary approach
    - Task 1.2: Modern approach
- Task 2: Multi-Camera Tracking (MCT)

#### Some example results

##### Task 1.2 Moder Approach Animation from the log files
<img src=https://github.com/mcv-m6-video/mcv-c6-2024-team6/assets/71498396/ec79bb83-8496-4500-8fa0-cb8bcd3d1fe2 width="400" height="auto">



## Week 5: Action Recognition
During this week we started with a new part of the project, which belongs to the University of Barcelona. We have worked with X3D-X6 model (the smallest one of the X3D family) in the HMDB51 Dataset.

We started working an improving the baseline training method given by the teacher. Then we added Multi-View Inference, first adding analysis by temporal windows and afterwards combining it with different spatial crops. Finally, we implemented the multi-view training strategy of [TSN](https://arxiv.org/pdf/1608.00859.pdf), along with some custom improvements.

#### Some images
##### Task 3: Temporal windows and different spatial crops

![task_3_pipeline](https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W5/example_imgs/task_3_pipeline.png)

![task_3_heatmaps](https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W5/example_imgs/heatmaps.png)

##### Task 4: TSN implementation and custom improvement

![TSN](https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W5/example_imgs/improvement_1.png)

![custom_improvements](https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W5/example_imgs/improvement_2.png)



## Week 6: Implementing alternative models 
During this week we had to change the previous week's model architectures to further improve the results. We tried very different implementations so to see which was the best working one.

Afterwards, we had to analyze the importance of temporal dynamics, proving if temporal information was necessary or not. Our work in this task was divided into two braches: shuffling the clips in order to look at the change in performance, and, on the other hand, using 2D nets to analyze each frame individually.

#### Bubble plot of the tried architectures in the first task:
![bubble_plot](https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W6/task_1/bubble_plot_slide.png)



## Week 7: Multimodality
The work of the final week was divided in two tasks: First we have to measure the performance of different modalities such as Optical flow, RGB difference and Skeleton extraction on their own. Afterwards, the second task consisted on mixing RGB information and alternative information. In this second task we analyzed different fussion methods: early fussion and late fussion.

#### Conclusions slide
![W7_conclusions](https://github.com/mcv-m6-video/mcv-c6-2024-team6/blob/main/W7/conclusion_slide.png)


