import cv2
import mediapipe as mp
import os
import time
import json
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

model_paths = {
    "pose_landmarker_full": "models/pose_landmarker_full.task",
    "pose_landmarker_heavy": "models/pose_landmarker_heavy.task",
    "pose_landmarker_lite": "models/pose_landmarker_lite.task"
}
video_dir = "data/videos"

def process_video(video_path, output_dir, model_name, model):
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_time = 0.0
    results = []


    while cap.isOpened():
        #print('Frame: ', video_path, frame_count)
        success, image = cap.read()
        if not success:
            break

        start_time = time.time()

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r = model.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if r.pose_landmarks:
            landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in r.pose_landmarks.landmark]
            results.append(landmarks)

        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        frame_count += 1

    avg_fps = frame_count / total_time

    path_parts = video_path.split("/")

    file_name = path_parts[-1].split(".")[0]

    npz_file = os.path.join(output_dir, f"{file_name}.npz")
    np.savez_compressed(npz_file, results=results)

    return avg_fps, inference_time

os.makedirs("data/results", exist_ok=True)
os.makedirs("data/npz", exist_ok=True)

metrics = {}
processed_metrics = {}

for model_name, model_path in model_paths.items():
    print(model_name)
    model = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5)

    output_dir = os.path.join("data/npz", model_name)
    os.makedirs(output_dir, exist_ok=True)
    metrics[model_name] = []

    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        avg_fps, inference_time = process_video(video_path, output_dir, model_name, model)
        metrics[model_name].append({"avg_fps": avg_fps, "inference_time": inference_time})
        print(video_dir, metrics[model_name])

    # Calculate and store average metrics after processing all videos for a model
    total_fps = sum(metric['avg_fps'] for metric in metrics[model_name])
    total_inf_time = sum(metric['inference_time'] for metric in metrics[model_name])
    avg_fps = total_fps / len(metrics[model_name])
    avg_inf_time = total_inf_time / len(metrics[model_name])

    processed_metrics[model_name] = {
        "avg_fps": avg_fps,
        "avg_inference_time": avg_inf_time
    }

    model.close()

with open("data/results/results.json", "w") as f:
    json.dump(processed_metrics, f, indent=4)
