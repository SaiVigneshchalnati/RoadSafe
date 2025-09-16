# seatbelt_detection.py

import sys
from ultralytics import YOLO
import imageio
import numpy as np
import os

def detect_seatbelt(video_path, output_path, model_path):
    # Load YOLOv8 model
    yolo_model = YOLO(model_path)

    # Read video frames
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps)

    # Run detection and save each annotated frame
    for frame in reader:
        results = yolo_model.predict(source=frame, conf=0.5)
        frame_with_boxes = results[0].plot()
        writer.append_data(np.array(frame_with_boxes))

    writer.close()
    print(f"\n✅ Processed video saved to: {output_path}")

if __name__ == "__main__":

    # Model path (fixed location)
    model_path = "/content/drive/MyDrive/DL Project/model/best_seatbelt.pt"

    video_path = "/content/traffic_compliance_outputs/filtered_output/processed_video(after_emergency).mp4"
    output_path = "/content/traffic_compliance_outputs/filtered_output/processed_video(after_seatbelt).mp4"

    # Run detection
    detect_seatbelt(video_path, output_path, model_path)
