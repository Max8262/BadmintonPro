import cv2
import os
import time
import psutil
import torch
from collections import deque
from ultralytics import YOLO

# Constants
MAX_NO_DETECTION_FRAMES = 10
FRAME_BUFFER_SIZE = 5
DETECTION_BUFFER_SIZE = 1
POST_DETECTION_BUFFER_SIZE = 5
RESIZE_FACTOR = 2
CONFIDENCE_THRESHOLD = 0.43
# Load YOLO model
model = YOLO("yolo11n_best.engine")

# Output folder
output_folder = 'Processed'
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture('demo.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
resize_width, resize_height = frame_width // RESIZE_FACTOR, frame_height // RESIZE_FACTOR

# Buffers and flags
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
detection_buffer = deque(maxlen=DETECTION_BUFFER_SIZE)
post_detection_buffer = deque(maxlen=POST_DETECTION_BUFFER_SIZE)
detection_flag = False
video_counter = 0
out_writer = None
detection_count = 0
no_detection_count = 0
frame_count = 0
start_time = time.time()

total_infer_time = 0

detected_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    small_frame = cv2.resize(frame, (resize_width, resize_height))
    
    # Measure inference time
    infer_start = time.time()
    results = model(small_frame)
    infer_time = time.time() - infer_start
    total_infer_time += infer_time
    
    detected = False
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0]
            if confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = x1 * RESIZE_FACTOR, x2 * RESIZE_FACTOR
                y1, y2 = y1 * RESIZE_FACTOR, y2 * RESIZE_FACTOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detected = True
                detected_frames += 1
    
    # Detection logic
    if detected:
        detection_count += 1
        no_detection_count = 0
        detection_buffer.append(frame)
        if detection_count >= 1 and not detection_flag:
            detection_flag = True
            video_counter += 1
            output_path = os.path.join(output_folder, f"real_time_part{video_counter}.mp4")
            out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        if detection_flag:
            while frame_buffer:
                out_writer.write(frame_buffer.popleft())
            out_writer.write(frame)
    else:
        if detection_flag:
            no_detection_count += 1
            post_detection_buffer.append(frame)
            if no_detection_count >= MAX_NO_DETECTION_FRAMES:
                while post_detection_buffer:
                    out_writer.write(post_detection_buffer.popleft())
                detection_flag = False
                detection_count = 0
                no_detection_count = 0
                if out_writer:
                    out_writer.release()
                    out_writer = None
        frame_buffer.append(frame)
    
cap.release()
if out_writer:
    out_writer.release()
cv2.destroyAllWindows()

# Final report
avg_infer_time = total_infer_time / frame_count if frame_count > 0 else 0
# Total processing time
total_time_elapsed = time.time() - start_time
actual_fps = frame_count / total_time_elapsed if total_time_elapsed > 0 else 0

print("\nFinal Report:")
print(f"Total Frames Processed: {frame_count}")
print(f"Detected Frames: {detected_frames}")
print(f"Total Processing Time: {total_time_elapsed:.2f} sec")
print(f"Average Inference Time per Frame: {avg_infer_time:.4f} sec")
print(f"Actual FPS: {actual_fps:.2f}")
print("Processing complete. Segments saved in:", output_folder)

