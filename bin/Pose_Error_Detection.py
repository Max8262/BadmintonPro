import cv2
import numpy as np
import math
import mediapipe as mp
import os
import subprocess
import time
import csv
import pandas as pd

errors = []
def analyze_badminton_poses_with_config(input_folder, config_file, ground_truth_folder=None, output_folder=None, display=True, delay=1, process_subset=None, skip_frames=0, max_frames=None):
    """
    Analyze badminton poses in videos from a folder using configuration file.
    
    Parameters:
    input_folder (str): Path to the folder containing input videos
    config_file (str): Path to CSV file with video names and pose types
    ground_truth_folder (str, optional): Path to folder containing ground truth CSV files
    output_folder (str, optional): Path to save output videos, if not specified no videos are saved
    display (bool): Whether to display the processing in real-time
    delay (int): Frame delay in milliseconds (controls playback speed)
    process_subset (list, optional): Only process specified video names, if None process all videos
    skip_frames (int): Number of frames to skip at the beginning
    max_frames (int, optional): Maximum number of frames to process, if None process entire video
    
    Returns:
    dict: Analysis results for each video
    """
    # Read configuration file
    try:
        # Try using Pandas
        pose_config = pd.read_csv(config_file)
        pose_dict = dict(zip(pose_config["Video Name"], pose_config["Final Majority Pose"]))
    except Exception as e:
        print(f"Cannot read CSV file with Pandas: {e}")
        try:
            # Try using basic CSV module
            pose_dict = {}
            with open(config_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "Video Name" in row and "Final Majority Pose" in row:
                        pose_dict[row["Video Name"]] = row["Final Majority Pose"]
                    else:
                        # If headers don't match, try to get first and third columns
                        keys = list(row.keys())
                        if len(keys) >= 3:
                            pose_dict[row[keys[0]]] = row[keys[2]]
        except Exception as e:
            print(f"Cannot read file with basic CSV module: {e}")
            return {"error": f"Cannot read configuration file: {e}"}
    
    # Ensure output folder exists
    if output_folder is None:
        # Default to 'analyzed_videos' folder in the current directory
        output_folder = "analyzed_videos"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Get all video files from the folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return {}
    
    # If processing subset is specified, filter the videos
    if process_subset:
        video_files = [f for f in video_files if f in process_subset]
        if not video_files:
            print(f"No matching videos found in the specified subset")
            return {}
    
    results_summary = {}
    
    # Initialize MediaPipe models
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing video: {video_file}")
        
        # Determine pose type
        pose_type = "clear"  # Default pose type
        if video_file in pose_dict:
            pose_type = pose_dict[video_file].lower()
            print(f"Using pose type from config: {pose_type}")
        else:
            print(f"Video {video_file} not found in config, using default: {pose_type}")
        
        # Automatically find the corresponding ground truth file based on pose type
        ground_truth_file = None
        if ground_truth_folder and pose_type:
            ground_truth_filename = f"allen_{pose_type}.csv"
            potential_path = os.path.join(ground_truth_folder, ground_truth_filename)
            if os.path.exists(potential_path):
                ground_truth_file = potential_path
                print(f"Using ground truth file: {ground_truth_file}")
            else:
                print(f"Warning: Cannot find ground truth file for pose type '{pose_type}': {potential_path}")
        
        # Set output path - always save the analyzed video
        output_file = f"analyzed_{video_file}"
        output_path = os.path.join(output_folder, output_file)
        
        # Analyze single video
        video_results = analyze_single_video(
            video_path, 
            ground_truth_file=ground_truth_file,
            output_path=output_path,  # Always provide output path
            pose_type=pose_type,
            display=display,
            delay=delay,
            mp_holistic=mp_holistic,
            mp_drawing=mp_drawing,
            skip_frames=skip_frames,
            max_frames=max_frames
        )
        
        # Add pose type to results
        video_results["pose_type"] = pose_type
        results_summary[video_file] = video_results
        
        print(f"Saved analyzed video to: {output_path}")
    
    return results_summary

def analyze_single_video(video_path, ground_truth_file=None, output_path=None, pose_type="clear", display=True, delay=1, 
                        mp_holistic=None, mp_drawing=None, skip_frames=0, max_frames=None):
    """
    Analyze badminton poses in a single video with option to compare with ground truth.
    
    Parameters:
    video_path (str): Path to the video file
    ground_truth_file (str, optional): Path to ground truth CSV file for skeleton comparison
    output_path (str, optional): Path to save the output video, if not specified no video is saved
    pose_type (str): Type of pose to analyze: "clear", "net", "drive", "serve", "lift"
    display (bool): Whether to display the processing in real-time
    delay (int): Frame delay in milliseconds (controls playback speed)
    mp_holistic, mp_drawing: MediaPipe objects, initialized if not provided
    skip_frames (int): Number of frames to skip at the beginning
    max_frames (int, optional): Maximum number of frames to process, if None process entire video
    
    Returns:
    dict: Video analysis results
    """
    
    # Initialize MediaPipe objects if not provided
    if mp_holistic is None:
        mp_holistic = mp.solutions.holistic
    if mp_drawing is None:
        mp_drawing = mp.solutions.drawing_utils
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return {"error": "Cannot open video"}
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer - always create if output_path is provided
    writer = None
    if output_path:
        try:
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"Warning: Could not initialize video writer for {output_path}")
                writer = None
        except Exception as e:
            print(f"Error creating video writer: {e}")
            writer = None
    
    # Track pose evaluation results
    metrics = {
        "P1_good": False,
        "P2_good": False,
        "P3_good": False,
        "P4_good": False,
        "P3_unconfirmed": False,
        "P1_high": False,
        "P1_low": False,
        "P2_high": False,
        "P2_low": False,
        "P1_count": 0,
        "P2_count": 0,
        "P3_count": 0,
        "P4_count": 0,
        "total_frames": 0,
        "frames_with_pose": 0,
        "errors": []  # List to store all errors
    }
    
    # Load ground truth data if provided
    ground_truth = None
    if ground_truth_file and os.path.exists(ground_truth_file):
        try:
            ground_truth = pd.read_csv(ground_truth_file)
            print(f"Loaded ground truth data: {ground_truth_file}, {len(ground_truth)} frames")
        except Exception as e:
            print(f"Error loading ground truth file: {e}")
            ground_truth = None
    
    # Initialize trajectories
    user_right_hand_trajectory = []
    pro_right_hand_trajectory = []
    
    count = 0
    processed_frames = 0
    frame_landmarks = []  # Store all frame landmarks for trajectory
    
    # Skip initial frames if requested
    for _ in range(skip_frames):
        ret, _ = cap.read()
        if not ret:
            print(f"Cannot skip specified frames, video may be too short")
            break
        count += 1
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as holistic:
        while cap.isOpened():
            count += 1
            ret, frame = cap.read()
            if not ret:
                break
                
            # Stop if maximum frames processed
            if max_frames is not None and processed_frames >= max_frames:
                break
                
            processed_frames += 1
            metrics["total_frames"] += 1
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Add video info
            overlay_text(image, f"Video: {os.path.basename(video_path)}", (50, 310))
            overlay_text(image, f"Pose type: {pose_type}", (50, 340))
            overlay_text(image, f"Frame: {count}/{total_frames}", (50, 370))
            
            if results.pose_landmarks:
                metrics["frames_with_pose"] += 1
                lmk = results.pose_landmarks.landmark
                
                # Extract landmarks for comparison and trajectory
                frame_landmark = []
                for landmark in lmk:
                    frame_landmark.extend([landmark.x, landmark.y])
                frame_landmarks.append(frame_landmark)
                
                # Store right hand position for trajectory
                right_hand_x = int(lmk[16].x * image.shape[1])
                right_hand_y = int(lmk[16].y * image.shape[0])
                user_right_hand_trajectory.append((right_hand_x, right_hand_y))
                
                # Draw detected pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
                
                if display:
                    overlay_landmark_numbers(image, lmk)
                
                # Compare with ground truth if available
                if ground_truth is not None and processed_frames <= len(ground_truth):
                    gt_idx = processed_frames - 1  # Adjust index as needed
                    gt_landmarks = ground_truth.iloc[gt_idx].values
                    
                    # Scale and translate ground truth landmarks to match detected pose
                    gt_landmarks = scale_ground_truth(gt_landmarks, frame_landmark, 12, 24)
                    gt_landmarks = translate_to_reference(gt_landmarks, frame_landmark, 24)
                    
                    # Draw ground truth pose (green)
                    draw_skeleton(image, gt_landmarks, color=(0, 255, 0), thickness=2)
                    
                    # Store ground truth right hand position for trajectory
                    if 16*2 < len(gt_landmarks):
                        gt_right_hand_x = int(gt_landmarks[16*2] * image.shape[1])
                        gt_right_hand_y = int(gt_landmarks[16*2+1] * image.shape[0])
                        pro_right_hand_trajectory.append((gt_right_hand_x, gt_right_hand_y))
                
                # Draw trajectories
                # Draw user's right hand trajectory (red)
                for i in range(1, len(user_right_hand_trajectory)):
                    cv2.line(image, user_right_hand_trajectory[i-1], user_right_hand_trajectory[i], (0, 0, 255), 2)
                
                # Draw professional's right hand trajectory (blue) if available
                for i in range(1, len(pro_right_hand_trajectory)):
                    cv2.line(image, pro_right_hand_trajectory[i-1], pro_right_hand_trajectory[i], (255, 0, 0), 2)
                
                # Add legend for trajectories
                if len(user_right_hand_trajectory) > 0:
                    overlay_text(image, "Your trajectory (Red)", (width - 250, 50), color=(0, 0, 255))
                if len(pro_right_hand_trajectory) > 0:
                    overlay_text(image, "Pro trajectory (Blue)", (width - 250, 80), color=(255, 0, 0))
                
                original_distance_12_24 = distance(lmk, 12, 24)
                scaled_distance_11_12 = 0
                if original_distance_12_24 > 0:
                    scale_factor = 1 / original_distance_12_24
                    original_distance_11_12 = distance(lmk, 11, 12)
                    scaled_distance_11_12 = original_distance_11_12 * scale_factor
                
                # Evaluate based on pose type
                if pose_type == "clear":
                    evaluate_clear_pose(image, lmk, metrics, scaled_distance_11_12)
                elif pose_type == "net":
                    evaluate_net_pose(image, lmk, metrics)
                elif pose_type == "drive":
                    evaluate_drive_pose(image, lmk, metrics)
                elif pose_type == "serve":
                    evaluate_serve_pose(image, lmk, metrics)
                elif pose_type == "lift":
                    evaluate_lift_pose(image, lmk, metrics)
                else:
                    print(f"Unknown pose type: {pose_type}, using clear pose evaluation")
                    
            
            # Always write to the video writer if it exists
            if writer:
                try:
                    writer.write(image)
                except Exception as e:
                    print(f"Error writing frame: {e}")
            
            # Display processing
            if display:
                cv2.imshow("Pose Analysis", image)
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Press 's' to save current frame
                    frame_dir = "saved_frames"
                    if not os.path.exists(frame_dir):
                        os.makedirs(frame_dir)
                    save_path = os.path.join(frame_dir, f"frame_{os.path.basename(video_path)}_{count}.jpg")
                    cv2.imwrite(save_path, image)
                    print(f"Saved frame to {save_path}")
    
    if pose_type == "clear":
        check_clear_pose_errors(metrics)
    elif pose_type == "net":
        check_net_pose_errors(metrics)
    elif pose_type == "drive":
        check_drive_pose_errors(metrics)
    elif pose_type == "serve":
        check_serve_pose_errors(metrics)
    elif pose_type == "lift":
        check_lift_pose_errors(metrics)
    else:
        print(f"Unknown pose type: {pose_type}, using clear pose evaluation")                

    # Release resources
    cap.release()
    if writer:
        writer.release()
        print(f"Finished writing video to {output_path}")
    if display:
        cv2.destroyAllWindows()
    
    # Collect errors - remove duplicate errors
    unique_errors = list(set(metrics.get("errors", [])))
    metrics["unique_errors"] = unique_errors
    
    return metrics

def draw(image, lmk, point_a, point_b):
    cv2.circle(image, (int(lmk[point_a].x * image.shape[1]), int(lmk[point_a].y * image.shape[0])), 10, (0, 0, 255), -1)
    cv2.circle(image, (int(lmk[point_b].x * image.shape[1]), int(lmk[point_b].y * image.shape[0])), 10, (0, 0, 255), -1)
    cv2.line(image, (int(lmk[point_a].x * image.shape[1]), int(lmk[point_a].y * image.shape[0])), (int(lmk[point_b].x * image.shape[1]), int(lmk[point_b].y * image.shape[0])), (0, 0, 255), 3)

def distance(lmk, a, b):
    """Calculate Euclidean distance between two landmark points"""
    ax, ay = lmk[a].x, lmk[a].y
    bx, by = lmk[b].x, lmk[b].y
    return math.sqrt((ax - bx)**2 + (ay - by)**2)

def angle(lmk, a, b, c):
    """Calculate angle formed by three points (B is the joint point)"""
    ax, ay = lmk[a].x, lmk[a].y
    bx, by = lmk[b].x, lmk[b].y
    cx, cy = lmk[c].x, lmk[c].y
    
    # Calculate vectors
    BA_x, BA_y = ax - bx, ay - by
    BC_x, BC_y = cx - bx, cy - by
    
    # Dot product and magnitude
    dot_product = BA_x * BC_x + BA_y * BC_y
    magnitude_BA = math.sqrt(BA_x**2 + BA_y**2)
    magnitude_BC = math.sqrt(BC_x**2 + BC_y**2)
    
    if magnitude_BA == 0 or magnitude_BC == 0:
        return None  # Avoid division by zero
    
    # Calculate angle (radians) and convert to degrees
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    # Clamp cos_angle to [-1, 1] to handle numerical errors
    cos_angle = max(-1, min(1, cos_angle))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def overlay_text(image, text, position, font_scale=0.7, color=(0, 255, 0), thickness=2):
    """Display text on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Add black background for better readability
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                 (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def evaluate_clear_pose(image, lmk, metrics, scaled_distance_11_12):

    """Evaluate clear shot pose - using original logic"""
    # Using original logic from the provided code

    if scaled_distance_11_12 > 0.35:
        metrics["P1_count"] += 1
    
    if angle(lmk, 13, 11, 23) >= 90:
        metrics["P2_count"] += 1

    if angle(lmk, 14, 12, 24) >= 90 and scaled_distance_11_12 > 0.39:
        metrics["P3_count"] += 1

    if angle(lmk, 11, 13, 15) >= 70:
        metrics["P4_count"] += 1

    if metrics["P1_count"] >= 4:
        metrics["P1_good"] = True
    
    if metrics["P2_count"] >= 4:
        metrics["P2_good"] = True
    
    if metrics["P3_count"] >= 4:
        metrics["P3_good"] = True
    
    if metrics["P4_count"] >= 5:
        metrics["P4_good"] = True
    

    p1_turned_sideways = metrics["P1_good"]
    
    if p1_turned_sideways:
        overlay_text(image, "Posture : Correct ", (50, 50), color=(0, 255, 0))  # Green for correct
  
    else:
        overlay_text(image, "Not Sideways, Need to be maintain a sideways position", (50, 50), color=(0, 0, 255))  # Red for error
        draw(image, lmk, 11, 12)
        draw(image, lmk, 11, 23)
        draw(image, lmk, 23, 24)
        draw(image, lmk, 24, 12)

    if not metrics["P1_good"]:
        if metrics["P3_good"]:
            overlay_text(image, "Can't Confirm if Right Hand Position is correct", (50, 150), color=(0, 255, 255))
            metrics["P3_unconfirmed"] = True
    else:
        if metrics["P3_good"]:
            overlay_text(image, "Right Hand : Correct", (50, 150), color=(0, 255, 0))
        else:
            overlay_text(image, "Right Hand is too Low, Need to be higher", (50, 150), color=(0, 0, 255))
            draw(image, lmk, 14, 16)
            draw(image, lmk, 14, 12)
    
    if metrics["P2_good"]:
        overlay_text(image, "Left Hand : Good Angle", (50, 100), color=(0, 255, 0))
    else:
        overlay_text(image, "Raise Your Left Hand", (50, 100), color=(0, 0, 255))
        draw(image, lmk, 11, 13)
        draw(image, lmk, 23, 11)

    
    if metrics["P4_good"]:
        overlay_text(image, "Left Hand : Good Posture", (50, 200), color=(0, 255, 0))
    else:
        overlay_text(image, "Stretch Your Left Hand", (50, 200), color=(0, 0, 255))
        draw(image, lmk, 11, 13)
        draw(image, lmk, 15, 13)

def check_clear_pose_errors(metrics):
    """Check errors for clear pose"""
    errors = []
    
    if metrics["P1_good"]:
        pass
    else:
        errors.append("側身不完全")
    
    if metrics["P2_good"]:
        pass
    else:
        errors.append("輔助手太低")
    
    if metrics["P3_good"] and metrics["P3_unconfirmed"] :
        pass
    else:
        errors.append("擊球手位置過低")

    
    if metrics["P4_good"]:
        pass
    else:
        errors.append("輔助手蜷縮")
    
    metrics["errors"].extend(errors)


def evaluate_net_pose(image, lmk, metrics):
    """Evaluate net shot pose - using original logic"""
    angle_12_14_16 = angle(lmk, 12, 14, 16)
    
    if angle_12_14_16 > 120:
        metrics["P1_count"] += 1

    if lmk[16].y < ((lmk[12].y * 2) + (lmk[24].y) * 1) / 3:
        metrics["P2_count"] += 1

    angle_24_26_28 = angle(lmk, 24, 26, 28)
    
    if angle_24_26_28 > 90 and lmk[26].x < lmk[32].x:
        metrics["P3_good"] = True
    else:
        metrics["P3_good"] = False
    
    if lmk[32].x < lmk[31].x:
        metrics["P4_good"] = False
    else:
        metrics["P4_good"] = True
    
    if metrics["P1_count"] >= 5:
        metrics["P1_good"] = True
    
    if metrics["P2_count"] >= 5:
        metrics["P2_good"] = True


    # Record errors in English
    errors = []
    
    # Display status with color-coding
    if metrics["P1_good"]:
        overlay_text(image, "Right Hand : Good Stretch", (50, 50), color=(0, 255, 0))
    else:
        overlay_text(image, "Need to stretch more on the right hand", (50, 50), color=(0, 0, 255))
        draw(image, lmk, 12, 14)
        draw(image, lmk, 16, 14)
    
    if metrics["P2_good"]:
        overlay_text(image, "Right Hand : Good Height", (50, 100), color=(0, 255, 0))
    else:
        overlay_text(image, "Please adjust your hand height", (50, 100), color=(0, 0, 255))
        draw(image, lmk, 12, 14)
        draw(image, lmk, 16, 14)
    
    if not metrics["P3_good"]:
        overlay_text(image, "Right Leg : Good Posture", (50, 150), color=(0, 255, 0))
    else:
        overlay_text(image, "Don't Overextend Your Knee. ", (50, 150), color=(0, 0, 255))
        draw(image, lmk, 24, 26)
        draw(image, lmk, 26, 28)
        draw(image, lmk, 28, 32)
    
    if metrics["P4_good"]:
        overlay_text(image, "Right Leg : Dominant Leg", (50, 200), color=(0, 255, 0))
    else:
        overlay_text(image, "Change dominant foot", (50, 200), color=(0, 0, 255))
        draw(image, lmk, 23, 25)
        draw(image, lmk, 25, 27)
        
def check_net_pose_errors(metrics):
    """Check errors for net pose"""
    errors = []
    
    if metrics["P1_good"]:
        pass
    else:
        errors.append("手臂蜷縮")
    
    if metrics["P2_good"]:
        pass
    else:
        errors.append("出手點過低")
    
    if metrics["P3_good"]:
        pass
    else:
        errors.append("膝蓋超伸")
    
    if metrics["P4_good"]:
        pass
    else:
        errors.append("無慣用腳在前")
    
    metrics["errors"].extend(errors)


def evaluate_drive_pose(image, lmk, metrics):
    """Evaluate drive shot pose - using original logic"""
    errors = []
    
    if lmk[0].y < lmk[16].y and lmk[16].y < ((lmk[12].y * 1) + (lmk[24].y * 1)) / 2:
        metrics["P1_count"] += 1
    
    if metrics["P1_count"] >= 5:
        metrics["P1_good"] = True
    
    if metrics["P1_good"]:
        overlay_text(image, "Right Hand : Good Posture", (50, 50), color=(0, 255, 0))

    elif lmk[0].y > lmk[16].y:
        overlay_text(image, "Lower Your Right Hand", (50, 50), color=(0, 0, 255))
        draw(image, lmk, 12, 14)
        draw(image, lmk, 16, 14)
        metrics["P1_high"] = True
    else:
        overlay_text(image, "Lift your Right hand", (50, 50), color=(0, 0, 255))
        draw(image, lmk, 12, 14)
        draw(image, lmk, 16, 14)
        metrics["P1_low"] = True

    
    # Calculate shoulder tilt angle
    if lmk[23].x != lmk[11].x:  # Prevent division by zero
        slope_12_24 = -1 * (lmk[23].y - lmk[11].y) / (lmk[23].x - lmk[11].x)
        angle_rad = math.atan(slope_12_24)
        angle_deg = math.degrees(angle_rad)
        
        if 70 <= angle_deg <= 84.8:
            metrics["P2_count"] += 1
        
        if metrics["P2_count"] >= 7:
            metrics["P2_good"] = True
    
    if metrics["P2_good"]:
        overlay_text(image, "Body : Good Posture", (50, 100), color=(0, 255, 0))
    else:
        overlay_text(image, "Adjust Your Body Tilt", (50, 100), color=(0, 0, 255))
        draw(image, lmk, 11, 12)
        draw(image, lmk, 11, 23)
        draw(image, lmk, 23, 24)
        draw(image, lmk, 24, 12)


def check_drive_pose_errors(metrics):
    """Check errors for drive pose"""
    errors = []
    if metrics["P1_good"]:
        pass
    elif metrics["P1_high"]:
        errors.append("球拍舉過高")
    else:
        errors.append("球拍未舉起")
    
    if metrics["P2_good"]:
        pass
    else:
        errors.append("身體直立或前傾")
    
    metrics["errors"].extend(errors)



def evaluate_serve_pose(image, lmk, metrics):
    """Evaluate serve pose - using original logic"""
    errors = []
    
    if lmk[32].x > lmk[31].x:
        metrics["P1_True"] = True
    else:
        metrics["P1_True"] = False
    
    if ((lmk[12].y * 2) + (lmk[24].y) * 1) / 3 < lmk[16].y and lmk[16].y < lmk[24].y:
        metrics["P2_count"] += 1
    

    if metrics["P2_count"] >= 4:
        metrics["P2_good"] = True

    if metrics["P1_good"]:
        overlay_text(image, "Right Foot : Dominant Foot", (50, 50), color=(0, 255, 0))
    else:
        overlay_text(image, "Change Your Dominant Foot", (50, 50), color=(0, 0, 255))
        draw(image, lmk, 23, 25)
        draw(image, lmk, 25, 27)
        draw(image, lmk, 24, 26)
        draw(image, lmk, 28, 26)
    
    if metrics["P2_good"]:
        overlay_text(image, "Right Hand : Good Position", (50, 100), color=(0, 255, 0))
    elif ((lmk[12].y * 2) + (lmk[24].y) * 1) / 3 > lmk[16].y:
        overlay_text(image, "Lower your right hand", (50, 100), color=(0, 0, 255))
        draw(image, lmk, 14, 12)
        draw(image, lmk, 14, 16)
        metrics["P2_low"] = True
    else:
        overlay_text(image, "Lift your right hand", (50, 100), color=(0, 0, 255))
        draw(image, lmk, 14, 12)
        draw(image, lmk, 14, 16)
        metrics["P2_high"] = True
        
def check_serve_pose_errors(metrics):
    """Check errors for serve pose"""
    errors = []
    
    if metrics["P1_good"]:
        pass
    else:
        errors.append("無慣用腳在前")
    
    if metrics["P2_good"]:
        pass
    elif metrics["P2_low"]:
        errors.append("發球位置過低")
    else:
        errors.append("發球位置過高")
    
    metrics["errors"].extend(errors)

def evaluate_lift_pose(image, lmk, metrics):
    """Evaluate lift shot pose - using original logic"""
    errors = []
    
    angle_24_26_28 = angle(lmk, 24, 26, 28)
    
    if lmk[32].x < lmk[31].x:
        metrics["P1_good"] = False
    else:
        metrics["P1_good"] = True
    
    if lmk[16].x < lmk[24].x:
        metrics["P2_good"] = False
    else:
        metrics["P2_good"] = True
    
    if angle_24_26_28 < 90 and lmk[26].x > lmk[32].x:
        metrics["P3_good"] = False
    else:
        metrics["P3_good"] = True
    
    if metrics["P1_good"]:
        overlay_text(image, "Right Foot : Dominant Foot", (50, 50), color=(0, 255, 0))

    else:
        overlay_text(image, "Change Dominant Foot", (50, 50), color=(0, 0, 255))
        draw(image, lmk, 23, 25)
        draw(image, lmk, 25, 27)
        draw(image, lmk, 24, 26)
        draw(image, lmk, 28, 26)
    
    if metrics["P2_good"]:
        overlay_text(image, "Right Hand : Good Position", (50, 100), color=(0, 255, 0))
    else:
        overlay_text(image, "Don't pull back your hand", (50, 100), color=(0, 0, 255))
        draw(image, lmk, 14, 12)
        draw(image, lmk, 14, 16)

    if metrics["P3_good"]:
        overlay_text(image, "Right Foot : Good Posture", (50, 150), color=(0, 255, 0))
    else:
        overlay_text(image, "Don't Overextend your Knee.", (50, 150), color=(0, 0, 255))
        draw(image, lmk, 24, 26)
        draw(image, lmk, 28, 26)
        draw(image, lmk, 28, 32)
        

def check_lift_pose_errors(metrics):
    """Check errors for lift pose"""
    errors = []
    
    if metrics["P1_good"]:
        pass
    else:
        errors.append("無慣用腳在前")
    
    if metrics["P2_good"]:
        pass
    else:
        errors.append("擊球手引拍過大")
    
    if metrics["P3_good"]:
        pass
    else:
        errors.append("膝蓋超伸")

    metrics["errors"].extend(errors)
    


def create_config_csv(input_folder, output_file='pose_config.csv'):
    """
    Create configuration file template from video folder.
    
    Parameters:
    input_folder (str): Path to the folder with input videos
    output_file (str): Path to save the output CSV file
    """
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return False
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video Name", "Frame Number", "Final Majority Pose"])
        for video in video_files:
            writer.writerow([video, "Final Majority", "clear"])  # Default pose type
    
    print(f"配置模板已創建：{output_file}")
    print("請編輯該文件以指定每個視頻的正確姿勢類型")
    return True

def generate_errors_report(results_summary, output_file='errors_report.csv'):
    """
    Generate a detailed report of all errors and save to CSV file.
    
    Parameters:
    results_summary (dict): Analysis results summary
    output_file (str): Path to save the output CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "影片名稱", 
            "姿勢類型", 
            "總幀數", 
            "檢測到姿勢的幀數",
            "所有錯誤（如視頻中顯示）"
        ])
        
        # Write results for each video
        for video, metrics in results_summary.items():
            if "error" in metrics:
                writer.writerow([video, "錯誤", 0, 0, metrics["error"]])
                continue
            
            # Get all errors (including duplicates to show frequency)
            all_errors = metrics.get("errors", [])
            unique_errors = metrics.get("unique_errors", [])
            
            # Count occurrences of each error
            error_counts = {}
            for error in unique_errors:
                error_counts[error] = all_errors.count(error)
            
            # Format errors with counts
            error_text = ""
            for error, count in error_counts.items():
                error_text += f"{error}; "
            
            if error_text:
                error_text = error_text[:-2]  # Remove trailing semicolon and space
            else:
                error_text = "沒有檢測到錯誤"
            
            writer.writerow([
                video,
                metrics.get("pose_type", "未知"),
                metrics.get("total_frames", 0),
                metrics.get("frames_with_pose", 0),
                error_text
            ])
    
    print(f"錯誤報告已生成：{output_file}")
    return True

# Functions for skeleton comparison and trajectory analysis
def translate_to_reference(gt_landmarks, extracted_landmarks, point_idx=24):
    """
    Adjust ground truth landmarks to match the extracted landmarks at the reference point.
    """
    # Extract coordinates for the reference point
    gt_point = np.array([gt_landmarks[point_idx * 2]+0.15, gt_landmarks[point_idx * 2 + 1]])
    extracted_point = np.array([extracted_landmarks[point_idx * 2], extracted_landmarks[point_idx * 2 + 1]])

    # Calculate translation vector
    translation = extracted_point - gt_point

    # Apply translation to ground truth landmarks
    translated_landmarks = []
    for i in range(0, len(gt_landmarks), 2):
        translated_landmarks.append(gt_landmarks[i] + translation[0])  # x translation
        translated_landmarks.append(gt_landmarks[i + 1] + translation[1])  # y translation

    return translated_landmarks

def scale_ground_truth(gt_landmarks, extracted_landmarks, point_1=12, point_2=24):
    """
    Scale the ground truth landmarks to match the distance between two reference points
    in the extracted landmarks.
    """
    # Ensure there are valid landmarks
    if np.any(np.isnan(gt_landmarks)) or np.any(np.isnan(extracted_landmarks)):
        return gt_landmarks  # Return unchanged if landmarks contain NaN

    # Calculate the distances between the two points in both ground truth and extracted landmarks
    gt_distance = np.sqrt(
        (gt_landmarks[point_2 * 2] - gt_landmarks[point_1 * 2]) ** 2 +
        (gt_landmarks[point_2 * 2 + 1] - gt_landmarks[point_1 * 2 + 1]) ** 2
    )
    extracted_distance = np.sqrt(
        (extracted_landmarks[point_2 * 2] - extracted_landmarks[point_1 * 2]) ** 2 +
        (extracted_landmarks[point_2 * 2 + 1] - extracted_landmarks[point_1 * 2 + 1]) ** 2
    )

    # Avoid division by zero
    if gt_distance == 0 or extracted_distance == 0:
        return gt_landmarks

    # Calculate the scaling factor
    scale_factor = extracted_distance / gt_distance

    # Scale all ground truth landmarks
    scaled_landmarks = []
    for i in range(0, len(gt_landmarks), 2):
        scaled_landmarks.append(gt_landmarks[i] * scale_factor)  # Scale x
        scaled_landmarks.append(gt_landmarks[i + 1] * scale_factor)  # Scale y

    return scaled_landmarks

def draw_line_between_keypoints(frame, landmarks, keypoint_1, keypoint_2, color=(255, 255, 255), thickness=2):
    """
    Draws a line between two specified keypoints on the frame.
    """
    if keypoint_1 * 2 < len(landmarks) and keypoint_2 * 2 < len(landmarks):
        x1, y1 = landmarks[keypoint_1 * 2], landmarks[keypoint_1 * 2 + 1]
        x2, y2 = landmarks[keypoint_2 * 2], landmarks[keypoint_2 * 2 + 1]

        # Ensure the landmarks are valid and not NaN
        if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
            # Scale normalized coordinates to pixel space
            x1, y1 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0])
            x2, y2 = int(x2 * frame.shape[1]), int(y2 * frame.shape[0])

            # Draw the line
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

def draw_skeleton(frame, landmarks, color=(0, 255, 0), thickness=2):
    """
    Draw a complete skeleton using the landmarks.
    """
    # Define the connections for drawing the skeleton
    connections = [
        (12, 14), (14, 16), (16, 22), (16, 20), (16, 18), (18, 20),
        (12, 24), (12, 11), (23, 24), (11, 23),
        (26, 24), (26, 28), (28, 32), (28, 30), (32, 30),
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (11, 13), (13, 15), (15, 21), (19, 15), (17, 15), (17, 19)
    ]
    
    # Draw each connection
    for point_1, point_2 in connections:
        draw_line_between_keypoints(frame, landmarks, point_1, point_2, color, thickness)
    
    # Draw circles for each landmark point
    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i + 1]
        if not np.isnan(x) and not np.isnan(y):
            x_px, y_px = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (x_px, y_px), 5, color, -1)

# Main program
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='羽球姿勢分析與骨架比較系統')
    parser.add_argument('--input', type=str, required=True, help='輸入視頻文件夾路徑')
    parser.add_argument('--config', type=str, help='配置文件路徑 (CSV)')
    parser.add_argument('--ground-truth', type=str, help='標準姿勢文件夾路徑 (CSV 文件)')
    parser.add_argument('--output', type=str, default="analyzed_videos", help='輸出視頻文件夾路徑')
    parser.add_argument('--report', type=str, default='errors_report.csv', help='錯誤報告輸出路徑')
    parser.add_argument('--create-config', action='store_true', help='創建配置文件模板')
    parser.add_argument('--no-display', action='store_false', dest='display', help='不顯示處理過程')
    parser.add_argument('--delay', type=int, default=1, help='幀延遲(毫秒)')
    parser.add_argument('--video', type=str, nargs='+', help='指定要處理的視頻名稱(可多個)')
    parser.add_argument('--skip-frames', type=int, default=0, help='跳過開始的幀數')
    parser.add_argument('--max-frames', type=int, default=None, help='最大處理幀數')
    
    args = parser.parse_args()
    
    # Create configuration template if requested
    if args.create_config:
        create_config_csv(args.input)
        exit(0)
    
    # Make sure output folder exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"創建輸出文件夾: {args.output}")
    
    # Use configuration file to analyze
    results = analyze_badminton_poses_with_config(
        args.input,
        args.config,
        ground_truth_folder=args.ground_truth,
        output_folder=args.output,  # Always provide output folder
        display=args.display,
        delay=args.delay,
        process_subset=args.video,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames
    )
    
    # Generate errors report
    generate_errors_report(results, args.report)
    
    # Print results summary
    print("\n===== 分析結果摘要 =====")
    print(f"分析後的視頻保存至: {args.output}")
    print(f"錯誤報告保存至: {args.report}")
    
    for video, metrics in results.items():
        print(f"\n視頻: {video}")
        if "error" in metrics:
            print(f"  錯誤: {metrics['error']}")
            continue
        
        print(f"  姿勢類型: {metrics.get('pose_type', '未知')}")
        print(f"  保存為: analyzed_{video}")
        
        # Print unique errors
        unique_errors = metrics.get("unique_errors", [])
        if unique_errors:
            print(f"  檢測到的錯誤 ({len(unique_errors)}):")
            for i, error in enumerate(unique_errors):
                print(f"    錯誤 {i+1}: {error}")
        else:
            print("  未檢測到錯誤。")
    

