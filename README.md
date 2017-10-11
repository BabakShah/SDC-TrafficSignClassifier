# SDC-TrafficSignDetection
Self-driving car project: Detecting Traffic Signs

# Self-Driving Car Project: Traffic Sign Classification

<img src="readme_images/SDC_TrafficSigns_Intro.jpg" width="480" alt="Mercedes-Benz Traffic Sign Detection" />

Hello there! I'm Babak. Let me introduce you to my second project in the self-driving car projects series. In this project I built a classifier to detect traffic signs on the road using convolutional nueral network with LaNet architecture, this is an important step in developing a self-driving car. Algorithm for this project is written in Python using TensorFlow, numpy, pickle and matplotlib libraries. The algorithm was trained on german traffic sign dataset and tested on actual traffic sign images.

**Contents**   

* IPython notebook
* Test_input folder
* Readme folder

### Pipeline:
The pipeline for detecting lane lines on the road is as follows:

1. Loading and visualizing the data.
2. Preprocessing the data (grayscale, normalize)
3. Designing and implementing a convolutional nueral network model with LeNet-5 architecture.
4. Training, validating and testing the model.
5. Testing the model on new traffic sign images.

Main stages of lane line detection can be seen below: 
<img src="readme_images/SDC_LaneLines_Stages.png" width="7500" alt="Six stages of lane line detection">
