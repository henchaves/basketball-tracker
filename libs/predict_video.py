import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import os
import math
from datetime import datetime
from moviepy.editor import VideoFileClip


VIDEO_SIZE = 32 * 30
TRAJECTORY = "on"

def load_model(model_path="models/final_model.pt"):
    model = YOLO(model_path)
    return model

def draw_point(image, coordinates, color=(0, 0, 255), radius=3):
    """Draws a point at the specified coordinates on the given image.

  Args:
    image: A NumPy array representing the image.
    coordinates: A NumPy array containing the coordinates of the point.
    color: A tuple of three integers representing the RGB color of the point.
    radius: The radius of the point in pixels.

  Returns:
    A NumPy array representing the image with the point drawn on it.
  """

    x, y = coordinates
    coord = (int(x), int(y))
    image = cv2.circle(image, coord, radius, color, thickness=-1)
    return image


def draw_line(image, coordinates_one, coordinates_two, color=(0, 0, 255), radius=3):
    """Draws a point at the specified coordinates on the given image.

  Args:
    image: A NumPy array representing the image.
    coordinates: A NumPy array containing the coordinates of the point.
    color: A tuple of three integers representing the RGB color of the point.
    radius: The radius of the point in pixels.

  Returns:
    A NumPy array representing the image with the point drawn on it.
  """

    x, y = coordinates_one
    coord_one = (int(x), int(y))
    x, y = coordinates_two
    coord_two = (int(x), int(y))
    image = cv2.line(image, coord_one, coord_two, color, 2)
    return image


def print_score(frame, shot_made):
    # Choose the font and other properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # White color

    text = "Made shot : " + str(shot_made)
    height, width = frame.shape[:2]

    # Calculate the position to place the text in the top right corner
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (width - text_size[0] - 10, 30)

    # Add the text to the image
    frame = cv2.putText(frame, text, text_position, font, font_scale, font_color, font_thickness)

    return frame



def predict_video(video_path):
    model = load_model()
    # video_paths = [video_path]

    all_shoot_made = []

    coords = []
    video_info = sv.VideoInfo.from_video_path(video_path)

    shoot_made = []
    trackers = [None] * 10
    coord_init_tracks = [None] * 10
    frame_init_tracks = [None] * 10

    actual_frame = [0]

    tracker = cv2.legacy.TrackerCSRT_create()
    is_tracking = [False]


    all_shoot_made.append(shoot_made)
    print(shoot_made)
    print(len(shoot_made))

    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"result_{time_str}.mp4"
    filename_output = f"static/{filename}"

    def process_frame(frame: np.ndarray, _) -> np.ndarray:
        actual_frame[0] += 1

        results = model.track(frame, imgsz=VIDEO_SIZE)[0]

        detections = sv.Detections.from_ultralytics(results)

        if results.boxes.id is not None:
            detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)

        # Annotate the different detected things
        box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.7, text_padding=1)

        labels = [f"#{tracker_id} " f"{model.names[class_id]} {confidence:0.2f}"
                  for _, _, confidence, class_id, tracker_id in detections]

        # Update the trackers
        for i, tracker in enumerate(trackers):
            if tracker is not None:
                (success, box) = tracker.update(frame)
                if success and frame_init_tracks[i] + 250 > actual_frame[0]:
                    (x, y, w, h) = [int(v) for v in box]
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                          (0, 255, 0), 2)
                    coord_init_tracks[i] = (x, y, x + w, y + h)
                else:
                    trackers[i] = None
                    coord_init_tracks[i] = None
                    frame_init_tracks[i] = None

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        for detection in detections:
            coord, _, confidence, class_id, _ = detection
            if class_id == 1:
                if len(shoot_made) == 0 or actual_frame[0] - shoot_made[-1] > 15:
                    shoot_made.append(actual_frame[0])

        frame = print_score(frame, len(shoot_made))

        for detection in detections:
            coord, _, confidence, class_id, _ = detection

            if class_id == 0:
                x0, y0, x1, y1 = coord

                w = x1 - x0
                h = y1 - y0
                # Create new tracker is necessary
                for i, tracker in enumerate(trackers):
                    one_close = False
                    box1 = [x0, y0, x1, y1]
                    for box2 in coord_init_tracks:
                        if box2 is not None:
                            center_x0, center_y0 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
                            center_x1, center_y1 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
                            # Calculate distance between centers
                            distance = math.sqrt((center_x1 - center_x0) ** 2 + (center_y1 - center_y0) ** 2)
                            if distance < 100:
                                one_close = True
                                break
                    if one_close:
                        break

                    if tracker is None:
                        trackers[i] = cv2.legacy.TrackerCSRT_create()
                        if w + h < 40 :
                            adapte_scale = (w + h) // 5
                        elif w + h < 65 :
                            adapte_scale = (w + h) // 8
                        else :
                            adapte_scale = (w + h) // 14


                        starting_bbox = [x0 - adapte_scale, y0 - adapte_scale, w + adapte_scale, h + adapte_scale]
                        trackers[i].init(frame, starting_bbox)
                        coord_init_tracks[i] = [x0, y0, x1, y1]
                        frame_init_tracks[i] = actual_frame[0]
                        break

        if TRAJECTORY == "on":

            id_to_skip = []
            # Get the different coords
            for detection in detections:
                coord, _, confidence, class_id, _ = detection
                if class_id == 0:
                    x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]

                    coord = (x1 + x2) / 2, (y1 + y2) / 2

                    if len(coords) == 0:
                        coords.append([])
                        coords[0].append(coord)
                    else:
                        x_new, y_new = coord
                        for i, coord_base in enumerate(coords):
                            x_base, y_base = coord_base[-1]
                            distance = np.linalg.norm(np.array([x_new, y_new]) - np.array([x_base, y_base]))
                            if distance < VIDEO_SIZE / 6:
                                coord_base.append(coord)
                            else:
                                coords.append([])
                                coords[-1].append(coord)
                            id_to_skip.append(i)

            for lines in coords:
                for i in range(len(lines) - 1):
                    if len(lines) > 1:
                        frame = draw_line(frame, lines[i], lines[i + 1])

            for point in coords:
                if len(point) == 1:
                    frame = draw_point(frame, point[0], radius=1)

            while len(coords) > 20:
                coords.pop(0)

        return frame

    sv.process_video(source_path=video_path, target_path=filename_output,
                     callback=process_frame)
    
    clip = VideoFileClip(filename_output)
    clip.write_videofile(filename_output, codec="libx264", audio_codec="aac")

    print(all_shoot_made)

    return filename


