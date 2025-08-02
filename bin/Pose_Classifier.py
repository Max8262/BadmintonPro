import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import os
import csv
from collections import Counter

# Load the pre-trained k-NN model, scaler, and label encoder
knn_model = joblib.load("knn_nor_t.pkl")  # Replace with your actual model path
scaler = joblib.load("knn_scaler.pkl")
label_encoder = joblib.load("knn_label_encoder.pkl")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5, model_complexity=0)

# Initialize MediaPipe Drawing for visualizing landmarks
mp_drawing = mp.solutions.drawing_utils

# Function for normalizing landmarks
def normalize_landmarks(pose_landmarks):
    # Convert the landmarks to a numpy array of x, y coordinates
    coordinates = np.array([[lm.x, lm.y] for lm in pose_landmarks])

    # Calculate the mean of the specified points (x11, x12, x23, x24) for centering
    x_mean = np.mean([coordinates[11, 0], coordinates[12, 0], coordinates[23, 0], coordinates[24, 0]])
    y_mean = np.mean([coordinates[11, 1], coordinates[12, 1], coordinates[23, 1], coordinates[24, 1]])

    # Center the coordinates by subtracting the mean, but exclude landmarks 1 to 10
    l = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    for i in l:
        coordinates[i, 0] -= x_mean
        coordinates[i, 1] -= y_mean

    # Return the centered coordinates as a flattened array, excluding landmarks 1 to 10
    filtered_coordinates = np.delete(coordinates, slice(1, 11), axis=0)  # Removing landmarks 1 to 10
    return filtered_coordinates.flatten()

# Path to folder containing videos
folder_path = "Processed"
total_frames = []

# Create CSV file to store results
def inf(video_folder):
    csv_file = video_folder + ".csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Video Name", "Frame Number", "Final Majority Pose"])

    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing video: {video_file}")

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            total_inference_time = 0
            predictions = []

            # Start measuring total processing time (including preprocessing)
            start_time = time.time()

            with mp_holistic.Holistic(static_image_mode=False, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5, 
                                      model_complexity=0) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Preprocessing step
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb_frame)

                    if results.pose_landmarks:
                        pose_landmarks = results.pose_landmarks.landmark
                        normalized_landmarks = normalize_landmarks(pose_landmarks)

                        if len(normalized_landmarks) == scaler.n_features_in_:
                            standardized_coordinates = scaler.transform([normalized_landmarks])
                            inference_start = time.time()  # Start of inference
                            predicted_class = knn_model.predict(standardized_coordinates)[0]
                            inference_end = time.time()  # End of inference

                            inference_time = inference_end - inference_start
                            total_inference_time += inference_time

                            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                            predictions.append(predicted_label)

                            # Annotate frame
                            cv2.putText(frame, f"Frame {frame_count}: {predicted_label}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        # cv2.imshow("hello", frame)

                    frame_count += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

            # Calculate total FPS (from preprocessing to end)
            elapsed_time = time.time() - start_time  # Total time taken from start to end
            fps = frame_count / elapsed_time  # Frames per second
            total_frames.append(frame_count)
            print(f"FPS for {video_file}: {fps:.2f}")


        if predictions:
            filtered_predictions = [pose for pose in predictions if pose.lower() not in ["other"]]
            
            if filtered_predictions:
                pose_counts = Counter(filtered_predictions).most_common()
                most_common_pose = pose_counts[0][0]  # Most common pose

                # Check if the second most common pose is "lift" and its count is greater than 6
                if len(pose_counts) > 1 and pose_counts[1][0].lower() == "lift" and pose_counts[1][1] > 6:
                    most_common_pose = "lift"
            else:
                most_common_pose = "Other"
            
            print(f"Final Predicted Pose for {video_file}: {most_common_pose}")
            
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([video_file, "Final Majority", most_common_pose])
        else:
            print(f"No pose detected for {video_file}")

    print("Processing complete.")





inf(folder_path)

