### Use Case

**Install**:
```angular2html
super-gradient
supervision
tqdm
ultralytics
```

#### Parser
```python
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    
    parser.add_argument(
        "--road_kind", default="S", choices=["C", "H", "S"], help="Modality of the road to keep the border cases", type=str
    )
```

if you want to add differents specifications in your custom video you can create a new key:value in the **specifications.py** file

finally just run

```python
python yolo/speed_estimation.py  --source_video data/vehicles.mp4 --road_kind H  --target_video_path data/results_Higway.mp4
```

