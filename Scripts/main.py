# old code

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from google.colab import files, drive
import torch
from IPython.display import display, HTML, Image
import pytesseract
from PIL import Image as PILImage
from ultralytics import YOLO
import supervision as sv
import sys

PROJECT_DIR = '/content/drive/MyDrive/DL Project/traffic_compliance_system'


# Function to upload traffic video
# def upload_video():
#     print("Please upload your traffic video...")
#     uploaded = files.upload()
#     video_path = list(uploaded.keys())[0]
#     print(f"Video '{video_path}' uploaded successfully!")
#     return video_path

# Load pre-trained YOLOv8 models
def load_models():
    # Main model for vehicle detection
    vehicle_model = YOLO('yolov8n.pt')

    # For better person detection (to count people on bikes)
    person_model = YOLO('yolov8n.pt')

    # For helmet detection (using general model initially)
    helmet_model = YOLO('yolov8n.pt')
    print("Using general purpose model for helmet detection. Fine-tuning recommended for better results.")

    #trying the lisence model
    # license_plate_model = YOLO('/content/best.pt')
    license_plate_model = YOLO('/content/drive/MyDrive/DL Project/model/best.pt')

    return vehicle_model, person_model, helmet_model, license_plate_model

# Helper functions

def class_id_to_name(class_id):
    classes = {
        0: "Person",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }
    return classes.get(class_id, "Vehicle")

def detect_lane_change(positions, lanes):
    # Simple lane change detection by checking if vehicle crossed lane boundaries
    crossed_lanes = False
    for i in range(1, len(positions)):
        for lane in lanes:
            lane_y = lane[0][1]
            if (positions[i-1] < lane_y and positions[i] > lane_y) or \
               (positions[i-1] > lane_y and positions[i] < lane_y):
                crossed_lanes = True
    return crossed_lanes

def estimate_speed(history, fps, frame_width, frame_height):
    # Simple speed estimation based on pixel movement
    if len(history) < 2:
        return 0

    # Get first and last position
    first_time, first_pos, _ = history[0]
    last_time, last_pos, _ = history[-1]

    # Calculate pixel distance moved
    pixel_distance = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)

    # Convert to time
    time_diff = last_time - first_time

    if time_diff == 0:
        return 0

    # Calculate speed (pixels per second)
    pixel_speed = pixel_distance / time_diff

    # Convert to km/h (rough approximation)
    conversion_factor = 0.1  # Example factor, needs calibration
    speed_kmh = pixel_speed * conversion_factor

    return speed_kmh

def count_riders_on_vehicle(vehicle_bbox, person_detections):
    """Count how many people are on a vehicle by checking overlap"""
    vx1, vy1, vx2, vy2 = vehicle_bbox

    # Minimum overlap percentage to consider a person as riding the vehicle
    min_overlap_pct = 0.5
    rider_count = 0

    for person_idx, (person_xyxy, _, _, _) in enumerate(zip(
        person_detections.xyxy, person_detections.confidence,
        person_detections.class_id, person_detections.tracker_id
    )):
        px1, py1, px2, py2 = map(int, person_xyxy)

        # Calculate intersection area
        ix1 = max(vx1, px1)
        iy1 = max(vy1, py1)
        ix2 = min(vx2, px2)
        iy2 = min(vy2, py2)

        if ix2 > ix1 and iy2 > iy1:  # There is an overlap
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            person_area = (px2 - px1) * (py2 - py1)

            # Calculate overlap percentage relative to person area
            overlap_pct = intersection_area / person_area

            if overlap_pct > min_overlap_pct:
                rider_count += 1

    return rider_count


def detect_helmet_violation(vehicle_bbox, person_detections, helmet_results):
    """Check if motorcycle riders are wearing helmets"""
    vx1, vy1, vx2, vy2 = vehicle_bbox

    riders_heads = []

    for person_idx, (person_xyxy, _, _, _) in enumerate(zip(
        person_detections.xyxy, person_detections.confidence,
        person_detections.class_id, person_detections.tracker_id
    )):
        px1, py1, px2, py2 = map(int, person_xyxy)

        # Calculate intersection with vehicle
        ix1 = max(vx1, px1)
        iy1 = max(vy1, py1)
        ix2 = min(vx2, px2)
        iy2 = min(vy2, py2)

        if ix2 > ix1 and iy2 > iy1:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            person_area = (px2 - px1) * (py2 - py1)

            overlap_pct = intersection_area / person_area

            if overlap_pct > 0.5:
                # Head region (top 1/4 of person bbox)
                head_y1 = py1
                head_y2 = py1 + (py2 - py1) // 4
                head_x1 = px1
                head_x2 = px2

                riders_heads.append((head_x1, head_y1, head_x2, head_y2))

    if not riders_heads:
        return False  # No riders => no violation

    # Parse helmet detections
    helmet_bboxes = []
    for det in helmet_results[0].boxes.xyxy:
        hx1, hy1, hx2, hy2 = map(int, det)
        helmet_bboxes.append((hx1, hy1, hx2, hy2))

    # Check each rider's head against helmet bboxes
    for head_bbox in riders_heads:
        hx1, hy1, hx2, hy2 = head_bbox
        has_helmet = False

        for helmet_bbox in helmet_bboxes:
            hhx1, hhy1, hhx2, hhy2 = helmet_bbox

            ix1 = max(hx1, hhx1)
            iy1 = max(hy1, hhy1)
            ix2 = min(hx2, hhx2)
            iy2 = min(hy2, hhy2)

            if ix2 > ix1 and iy2 > iy1:
                has_helmet = True
                break

        if not has_helmet:
            return True  # Violation found (no helmet)

    return False  # All heads have helmets => no violation

import cv2

def helmet_detection(img, model, class_names=['helmet', 'mobile_usage', 'no_helmet', 'triple_ride'], conf_threshold=0.3):
    """
    Detects helmet-related violations in a given image/frame using a pre-loaded YOLOv5 model.

    Args:
        img (np.array): Image or frame (BGR format).
        model: Preloaded YOLOv5 model.
        class_names (list): Class labels corresponding to the model's training.
        conf_threshold (float): Minimum confidence to display detections.

    Returns:
        img (np.array): Image with drawn bounding boxes.
        detections (list): List of detected class names.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    df = results.pandas().xyxy[0]

    detections = []

    for _, row in df.iterrows():
        if row['confidence'] > conf_threshold:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            detections.append(row['name'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, detections


def extract_license_plate(frame, vehicle_bbox, license_plate_model):
    """Detect and extract license plate using trained YOLO model"""
    x1, y1, x2, y2 = vehicle_bbox
    vehicle_region = frame[y1:y2, x1:x2]

    # Run inference using the license plate detector
    results = license_plate_model.predict(vehicle_region, imgsz=640, conf=0.4)  # Adjust conf/imgsz if needed

    # Extract the first plate detection (assumes 1 plate per vehicle)
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # Assume highest confidence box
            best_box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            px1, py1, px2, py2 = best_box

            # Extract plate from vehicle_region and convert to full frame coords
            plate_img = vehicle_region[py1:py2, px1:px2]

            # OCR using pytesseract
            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                plate_text = pytesseract.image_to_string(
                    plate_thresh, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                plate_text = ''.join(c for c in plate_text if c.isalnum())

                if len(plate_text) >= 4:
                    return plate_text
            except Exception as e:
                print(f"OCR error: {e}")

    return None


def clean_vehicle_history(vehicle_history, current_frame, fps):
    # Remove history older than 5 seconds to save memory
    time_threshold = (current_frame / fps) - 5

    for vehicle_id in list(vehicle_history.keys()):
        # Filter to keep only recent positions
        vehicle_history[vehicle_id] = [
            pos for pos in vehicle_history[vehicle_id]
            if pos[0] > time_threshold
        ]

        # Remove vehicles not seen recently
        if not vehicle_history[vehicle_id]:
            del vehicle_history[vehicle_id]



# Main video processing function
def process_video(video_path, vehicle_model, person_model, helmet_model, license_plate_model, sampling_rate=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare for video saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = f"{PROJECT_DIR}/processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps//sampling_rate, (width, height))

    # Initialize trackers
    vehicle_tracker = sv.ByteTrack()
    person_tracker = sv.ByteTrack()

    # Initialize violation database
    violations_db = []

    # Define traffic lane boundaries (example, adjust for your video)
    lanes = [
        [(0, height//2), (width, height//2)],  # Middle line
        [(0, height//4), (width, height//4)],  # Upper line
        [(0, 3*height//4), (width, 3*height//4)]  # Lower line
    ]

    # Vehicle position history for speed and lane change detection
    vehicle_history = {}

    # Process frames
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process only every few frames for efficiency
        if frame_idx % sampling_rate != 0:
            frame_idx += 1
            continue

        # Make a copy of the frame for processing
        processed_frame = frame.copy()

        # Detect vehicles
        vehicle_results = vehicle_model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
        vehicle_detections = sv.Detections.from_ultralytics(vehicle_results[0])
        vehicle_detections = vehicle_tracker.update_with_detections(vehicle_detections)

        # Detect people for counting riders
        person_results = person_model(frame, classes=[0])  # person
        person_detections = sv.Detections.from_ultralytics(person_results[0])
        person_detections = person_tracker.update_with_detections(person_detections)

        # Detect helmets
        helmet_results = helmet_model(frame)

        # Process each vehicle detection
        for detection_idx, (xyxy, confidence, class_id, tracker_id) in enumerate(zip(
            vehicle_detections.xyxy, vehicle_detections.confidence, vehicle_detections.class_id, vehicle_detections.tracker_id
        )):
            if tracker_id is None:
                continue

            # Calculate vehicle position and size
            x1, y1, x2, y2 = map(int, xyxy)
            vehicle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            vehicle_size = (x2 - x1) * (y2 - y1)

            # Store vehicle position history
            if tracker_id not in vehicle_history:
                vehicle_history[tracker_id] = []
            vehicle_history[tracker_id].append((frame_idx/fps, vehicle_center, vehicle_size))

            # Check for violations
            violations = []

            # Vehicle type
            vehicle_type = class_id_to_name(class_id)

            # 1. Lane change detection
            if len(vehicle_history[tracker_id]) > 5:
                recent_positions = [pos[1][1] for pos in vehicle_history[tracker_id][-5:]]
                lane_change = detect_lane_change(recent_positions, lanes)
                if lane_change:
                    violations.append("Lane Change")

            # 2. Speeding detection (simplified)
            if len(vehicle_history[tracker_id]) > 10:
                speed = estimate_speed(vehicle_history[tracker_id][-10:], fps, width, height)
                if speed > 50:  # example threshold
                    violations.append(f"Speed: {speed:.1f}km/h")

            # 3. For motorcycles, check for multiple riders and helmet violations
            if vehicle_type == "Motorcycle":
                # Count people on this motorcycle by checking overlap
                motorcycle_bbox = [x1, y1, x2, y2]
                riders = count_riders_on_vehicle(motorcycle_bbox, person_detections)

                if riders > 2:  # More than 2 people on motorcycle
                    violations.append(f"Multiple Riders ({riders})")

                # Check for helmets on motorcycle riders
                helmet_violation = detect_helmet_violation(motorcycle_bbox, person_detections, helmet_results)
                if helmet_violation and riders > 0:
                    violations.append("No Helmet")

            # 4. Extract license plate for violating vehicles
            if violations:
                plate_text = extract_license_plate(frame, (x1, y1, x2, y2), license_plate_model)
                if plate_text:
                    violations.append(f"Plate: {plate_text}")

                # Record the violation
                violations_db.append({
                    "frame_idx": frame_idx,
                    "time": frame_idx/fps,
                    "tracker_id": int(tracker_id),
                    "vehicle_type": vehicle_type,
                    "violations": ", ".join(violations),
                    "license_plate": plate_text if plate_text else "Unknown",
                    "confidence": float(confidence)
                })

                # Save violation snapshot
                violation_img = frame.copy()
                cv2.rectangle(violation_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(violation_img, ", ".join(violations),
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                snapshot_path = f"{PROJECT_DIR}/detected_violations/violation_{frame_idx}_{int(tracker_id)}.jpg"
                cv2.imwrite(snapshot_path, violation_img)

            # Annotate vehicles on frame
            label = f"ID:{tracker_id} {vehicle_type}"
            if violations:
                label += f" VIOLATION: {', '.join(violations)}"
                color = (0, 0, 255)  # Red for violations
            else:
                color = (0, 255, 0)  # Green for compliant vehicles

            # Draw vehicle bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw traffic lanes
        for lane in lanes:
            cv2.line(processed_frame, lane[0], lane[1], (255, 255, 0), 2)

        # Save annotated frame
        if frame_idx % (sampling_rate * 10) == 0:  # Save every 10th processed frame
            frame_path = f"{PROJECT_DIR}/processed_frames/frame_{frame_idx}.jpg"
            cv2.imwrite(frame_path, processed_frame)

        # Write frame to output video
        out.write(processed_frame)

        # Clean up old history to save memory
        if frame_idx % 100 == 0:
            clean_vehicle_history(vehicle_history, frame_idx, fps)

        frame_idx += 1

        # # Show progress
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames out of {frame_count} ({frame_idx/frame_count*100:.1f}%)")

    # Clean up
    cap.release()
    out.release()

    # Save violations database
    violations_df = pd.DataFrame(violations_db)
    if not violations_df.empty:
        violations_df.to_csv(f"{PROJECT_DIR}/violations_report.csv", index=False)
    else:
        print("No violations detected.")
        violations_df = pd.DataFrame(columns=["frame_idx", "time", "tracker_id", "vehicle_type", "violations", "license_plate", "confidence"])
        violations_df.to_csv(f"{PROJECT_DIR}/violations_report.csv", index=False)

    print(f"Processing complete! Output video saved to {output_path}")
    print(f"Detected {len(violations_db)} violations")

    return violations_df, output_path





# Create dashboard for visualization
def create_violations_dashboard(violations_df):
    if violations_df.empty:
        print("No violations detected.")
        return

    # Basic statistics
    total_violations = len(violations_df)
    vehicle_types = violations_df['vehicle_type'].value_counts()
    violation_types = []

    for violations_str in violations_df['violations']:
        for violation in violations_str.split(', '):
            if not violation.startswith('Plate:'):
                violation_types.append(violation)

    violation_counts = pd.Series(violation_types).value_counts()

    # Generate plots
    plt.figure(figsize=(15, 10))

    # Vehicle type distribution
    plt.subplot(2, 2, 1)
    vehicle_types.plot(kind='bar')
    plt.title('Violations by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Count')

    # Violation type distribution
    plt.subplot(2, 2, 2)
    violation_counts.plot(kind='bar')
    plt.title('Violation Types')
    plt.xlabel('Violation Type')
    plt.ylabel('Count')

    # Violations over time
    plt.subplot(2, 2, 3)
    violations_df['time_bin'] = pd.cut(violations_df['time'], bins=10)
    time_distribution = violations_df.groupby('time_bin').size()
    time_distribution.plot(kind='line')
    plt.title('Violations Over Time')
    plt.xlabel('Time in Video (seconds)')
    plt.ylabel('Count')

    # Add specific section for bike-related violations
    plt.subplot(2, 2, 4)
    bike_violations = [v for v in violation_types if v.startswith("Multiple Riders") or v == "No Helmet"]
    if bike_violations:
        bike_violation_counts = pd.Series(bike_violations).value_counts()
        bike_violation_counts.plot(kind='bar')
        plt.title('Motorcycle-Specific Violations')
        plt.xlabel('Violation Type')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, 'No motorcycle-specific violations detected',
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Motorcycle-Specific Violations')

    # Save dashboard
    plt.tight_layout()
    dashboard_path = f"{PROJECT_DIR}/violations_dashboard.png"
    plt.savefig(dashboard_path)
    plt.close()

    # Display dashboard
    display(Image(filename=dashboard_path))

    # Create an HTML report
    report = f"""
    <h1>Traffic Compliance System - Violation Report</h1>
    <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Summary</h2>
    <p>Total violations detected: {total_violations}</p>

    <h2>Violation Details</h2>
    <table border="1">
    <tr>
        <th>Time (s)</th>
        <th>Vehicle Type</th>
        <th>Violations</th>
        <th>License Plate</th>
    </tr>
    """

    for _, row in violations_df.iterrows():
        report += f"""
        <tr>
            <td>{row['time']:.2f}</td>
            <td>{row['vehicle_type']}</td>
            <td>{row['violations']}</td>
            <td>{row['license_plate']}</td>
        </tr>
        """

    report += "</table>"

    # Save HTML report
    html_path = f"{PROJECT_DIR}/violations_report.html"
    with open(html_path, 'w') as f:
        f.write(report)

    print(f"Violation dashboard saved to {dashboard_path}")
    print(f"Detailed report saved to {html_path}")

    return dashboard_path, html_path




# Main execution
def main(video_path):
    print("Enhanced Traffic Compliance System - Deep Learning Project")
    print("=" * 50)
    print("This system can detect:")
    print("1. Standard traffic violations (speeding, lane changes)")
    print("2. Multiple riders on motorcycles (>2 people)")
    print("3. Riders without helmets")
    print("4. License plate recognition")
    print("=" * 50)

    # Step 1: Upload video
    # video_path = upload_video()
    # video_path = "/content/drive/MyDrive/Uploaded videos/traffic15_seatbelt.mp4"

    # Step 2: Load YOLO models
    vehicle_model, person_model, helmet_model, license_plate_model = load_models()

    # Step 3: Process video to detect violations
    print("Processing video for violations...")
    violations_df, output_path = process_video(video_path, vehicle_model, person_model, helmet_model, license_plate_model)

    # Step 4: Create visualization dashboard
    print("Creating violations dashboard...")
    create_violations_dashboard(violations_df)

    # Step 5: Display results
    print("\nProject Results Summary:")
    print(f"- Processed video saved to: {output_path}")
    print(f"- Detected {len(violations_df)} violations")
    print(f"- Violation records saved to: {PROJECT_DIR}/violations_report.csv")
    print(f"- Violation snapshots saved to: {PROJECT_DIR}/detected_violations/")

# Run the main program
if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        raise ValueError("❌ No video path provided! Usage: python main.py <video_path>")

    main(video_path)