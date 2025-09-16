import torch
from IPython.display import Video

# Define class names
class_names = ['helmet', 'mobile_usage', 'no_helmet', 'triple_ride']  # Modify if your dataset has more/different classes

# Get the index of "triple_ride"
# class_to_filter = 'triple_ride'
# class_index = class_names.index(class_to_filter)

# Run YOLOv5 on the uploaded video and filter only "triple_ride"
!python /content/yolov5/detect.py --weights "/content/drive/MyDrive/DL Project/model/best_3rider.pt" \
                  --source "/content/drive/MyDrive/DL Project/traffic_compliance_system/processed_video.mp4" \
                  --conf 0.3 \
                  --save-txt \
                  --save-conf \
                  --project /content/traffic_compliance_outputs \
                  --name filtered_output \
                  --exist-ok
