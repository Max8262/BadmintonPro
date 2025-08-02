# BadmintonPro
Badminton Pro is a software dedicated to the training and improvements of badminton amateurs. With the increase of people being interested in badminton, we propose a software for amateurs to improve. Badminton Pro is defined to have several features, we would like the user to just film a long video, then Badminton Pro would auto-edit the long videos to small clips, these clips would have to be a swing of the racket. Then Badminton would classify what just happened in this clip alone. Finally, we would use a rule-based model using a HPE(Human Pose Estimation Model) to figure out what pose error was present in the clip. (We would also want this system architecture to be expanded to other sports as well : Tennis, Squatch, and so on…….). In this Repo, we will be teaching you how to make your own badminton assistant.

### ‼️ATTENTION‼️
Note: This is only an idea on how to make an app to “maybe” improve your badminton skills. This hasn’t been entirely proven to improve one’s skills.
This is a medium level project that is a great start for Deep Learning Integration projects, learning AI-related models wouldn't be that hard after this project.

### 📔 Table of Contents
1. Introduction & Demo
2. Common Badminton Poses
3. Data Collection
4. Pose Classifiers
5. YOLO Training
6. Common Metrics
7. Pose Error Detection
8. OpenAI Integration

# 1.Introduction & Demo
Link: 
Abstract: This course provides an introduction to Badminton Pro, covering our current development progress and achievements. We'll explore the project's objectives, examine the repository contents, discuss implementation constraints, and conclude with a live demonstration of the application's functionality.

## 🎯 Learning Goals
1. How to run and choose HPE models.
2. How to write image processing tools with opencv.
3. How to run, train, and evaluate YOLO models.
4. How to make&compare your own pose classifiers.
5. How to make a web html file to import videos and output video results.

## 💾 In this Repo......
In this repo, we will be using YOLO & three different classifiers to accomplish three goals.
1.  Develop a procedure to auto-edit badminton long videos into clips.
2.  Run pose classification on these clips to determine what the clips' pose was.
3.  Determine what pose errors were according to the clips' determined pose.
4.  (Optional) Use LLM API's to integrate pose error and pose to give pose correction advices.

## ⛔ Prerequisites
1. Is an amateur who is interested in playing badminton, but doesn't know where to start.
2. Has purchased a camera stand that is at least 1.5 meters above.
3. Has a phone that can record half a badminton ground range, with the person's full body being recorded. 

## ⏩ QuickStart
First, install the required python pip packages.
```

pip install -r requirements.txt
```
You can run the `index_en.html` and the `server.py` for a quick preview of what we've acheived. There would be a small video for you to download and try it out.
(Video is inside repo)

## 📦 Packages & Environment Introduction
Python Version 3.10.9
1.	flask (v3.1.0): Provides lightweight web framework functionality. 
2.	joblib (v1.4.2): Used for serializing and saving machine learning models. 
3.	matplotlib (v3.10.0): Responsible for plotting experimental result charts. 
4.	mediapipe (v0.10.20): Human pose keypoint tracking library. 
5.	moviepy (v1.0.3): Handles video slow-motion processing. 
6.	numpy (v1.26.4): Mathematical computation library for data centralization and processing. 
7.	opencv-python (v4.11.0.86): Manages video reading, segmentation, and preprocessing. 
8.	openai (v1.61.0): Enables connection to ChatGPT-4 for pose improvement suggestions. 
9.	pandas (v2.2.3): Handles CSV file reading and data manipulation. 
10.	scikit-learn (v1.6.1): Open-source machine learning library for generating model performance reports. 
11.	seaborn (v0.13.2): Responsible for confusion matrix visualization. 
12.	tensorflow (v2.18.0): Open-source deep learning framework for model training. 
13.	tensorrt (v10.8.0.43): Handles YOLO model quantization and optimization. 
14.	ultralytics (v8.3.70): YOLOv11 object detection model library.

# 2. Common Badminton Poses

# 3. Data Collection

# 4. Pose Classifiers

# 5. YOLO Training

# 6. Common Metrics

# 7. Pose Error Detection

# 8. OpenAI Integration














