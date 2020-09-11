# yolo-webcam
Runs yolo v3-tiny on the device camera. Just run the script with default parameters, or look below for example on how to run with custom parameter values.
Python version used is python 3.6. Probably works on other python 3 versions, but is not tested. I recommend using Anaconda3 as
enviroment manager (and interpreter). It is set to display resolution 1440x900, this is scalable with the RESOLUTION_FACTOR parameter. Press **q** to
quit the video stream and end the program.
# Requirements
* Python 3.6 (3.x?) (conda create -n "env-name" python=3.6 -> conda activate env-name)
* cv2 (pip install opencv-python)
* numpy (pip install numpy)

If you want to run other versions of yolo you have to download the weights and config files from [here](https://pjreddie.com/darknet/yolo/) and
adapt the script to target those files instead of the yolov3-tiny version.
# How to run
```
python python-yolo-cam.py -ct=0.8 -ot=0.9 -rs=0.5
```
This will run the streaming with confidence theshold of 0.8, overlap threshold of 0.9 and scale the 1440x900 resolution by 0.5 (50%).
```
python python-yolo-cam.py
```
This will run with default parameters.
