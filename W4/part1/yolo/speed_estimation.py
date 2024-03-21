from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv
import CSVSink as csvs

from parser import parse_arguments

import matplotlib.pyplot as plt

import copy

import specifications as spec




class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)




if __name__ == "__main__":
    args = parse_arguments()
    
    ## This implies the assumptios of meters and the zone where to keep the perspective
    SOURCE = spec.ROADS[args.road_kind]["SOURCE"]
    TARGET_WIDTH = spec.ROADS[args.road_kind]["TARGET_WIDTH"]
    TARGET_HEIGHT = spec.ROADS[args.road_kind]["TARGET_HEIGHT"]
    
    TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
    )
    
    CLASS_SPECIFICATIONS = spec.ROADS[args.road_kind]["CLASSES"]
    MAX_SPEED = spec.ROADS[args.road_kind]["MAX_SPEED"]


    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO("yolov8x.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_thresh=args.confidence_threshold
    )

    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )


    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(
        polygon=SOURCE, frame_resolution_wh=video_info.resolution_wh
    )
    
    zone_annotator = sv.PolygonZoneAnnotator(
            zone=polygon_zone,
            color=sv.Color.RED,
            thickness=2,
            text_thickness=2,
            text_scale=0.5
    )
    

    CSVsink = csvs.CSVSink(f"data/{args.road_kind}_content.csv")


    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    with CSVsink as csvsink:
        with sv.VideoSink(args.target_video_path, video_info) as sink:
            for idx, frame in enumerate(frame_generator):
                #if idx < 19000:
                #    continue

                print(idx)
                photo = False
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > args.confidence_threshold]                    
                detections = detections[polygon_zone.trigger(detections)]
                detections = detections.with_nms(threshold=args.iou_threshold)
                detections = byte_track.update_with_detections(detections=detections)
                
                total_objects_zone = len(detections.class_id)
                
                cars_going_top = 0
                cars_going_down = 0               
                
                points = detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
                
                points = view_transformer.transform_points(points=points).astype(int)
                csvsink.append(detections, custom_data={'frame_id':idx, "xy":points})

                for _, (tracker_id, [_, y]) in enumerate(zip(detections.tracker_id, points)):
                    coordinates[tracker_id].append(y)
                    
                labels = []
                colors = []

                for ii, tracker_id in enumerate(detections.tracker_id):

                    if detections.class_id[ii] not in CLASS_SPECIFICATIONS.keys():
                        total_objects_zone -= 1
                        continue
                    
                    if len(coordinates[tracker_id]) < video_info.fps / 2:
                        labels.append(f"#{tracker_id}")
                        colors.append(spec.LOOKUP_COLORS[detections.class_id[ii]])

                    else:
                        coordinate_start = coordinates[tracker_id][-1]
                        coordinate_end = coordinates[tracker_id][0]
                        
                        direction = coordinate_start - coordinate_end
                        
                        distance = abs(direction)
                        time = len(coordinates[tracker_id]) / video_info.fps
                        speed = distance / time * 3.6
                        if speed > MAX_SPEED:
                            labels.append(f"#{tracker_id} {int(speed)} km/h: Dangerous")
                            colors.append(6)
                            photo = True
                            if direction< 0:
                                cars_going_top += 1
                            else:
                                cars_going_down += 1
                        
                        else:
                            if speed < 5:
                                labels.append(f"#{tracker_id} parking? (-1)")

                            else:
                                labels.append(f"#{tracker_id} {int(speed)} km/h")
                                if direction< 0:
                                    cars_going_top += 1
                                else:
                                    cars_going_down += 1
                                
                            colors.append(spec.LOOKUP_COLORS[detections.class_id[ii]])
                            
                            
                annotated_frame = frame.copy()
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                annotated_frame = bounding_box_annotator.annotate(
                    scene=annotated_frame, detections=detections, custom_color_lookup=np.array(colors)
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
                
                annotated_frame = zone_annotator.annotate(scene=annotated_frame, label=f"trigger zone | Total_ROI={total_objects_zone} | T={cars_going_top} | D={cars_going_down}")
                sink.write_frame(annotated_frame)
            
                if photo:
                    cv2.imwrite(f"snapshots/{args.road_kind}_speed_instance_{idx}.png", annotated_frame)
                    
                    
                if idx == 998: exit()

                #cv2.imshow("frame", annotated_frame)
                #if cv2.waitKey(1) & 0xFF == ord("q"):
                #    break
            #cv2.destroyAllWindows()