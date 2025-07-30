# BadmintonPro
Badminton Pro is a software dedicated to the training and improvements of badminton amateurs. With the increase of people being interested in badminton, we propose a software for amateurs to improve. Badminton Pro is defined to have several features, we would like the user to just film a long video, then Badminton Pro would auto-edit the long videos to small clips, these clips would have to be a swing of the racket. Then Badminton would classify what just happened in this clip alone. Finally, we would use a rule-based model using a HPE(Human Pose Estimation Model) to figure out what pose error was present in the clip. (We would also want this system architecture to be expanded to other sports as well : Tennis, Squatch, and so on…….). In this Repo, we will be teaching you how to make your own badminton assistant. 

## Please Read This First......
Note: This is only an idea on how to make an app to “maybe” improve your badminton skills. This hasn’t been entirely proven to improve one’s skills.
This is a medium level project that is a great start for Deep Learning Integration projects, learning AI-related models wouldn't be that hard after this project.

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
Here's the list of packages that we would be using today.
- Mediapipe
- TensorRT
- Ultralytics
- 
