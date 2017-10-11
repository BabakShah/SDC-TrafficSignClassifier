# SDC-TrafficSignDetection
Self-driving car project: Detecting Traffic Signs

# Self-Driving Car Project: Traffic Sign Classification

<img src="readme_images/SDC_TrafficSigns_Intro.jpg" width="480" alt="Mercedes-Benz Traffic Sign Detection" />

Hello there! I'm Babak. Let me introduce you to my second project in the self-driving car projects series. In this project traffic signs on the road are automatically detected and classified using convolutional nueral network with LaNet architecture, this is an important step in developing a self-driving. Algorithm for this project is written in Pyhton using TensorFlow, numpy, pickle and matplotlib libraries. The algorithm was tested on actual traffic sign images.

**Contents**   

* IPython notebook
* Test_input folder
* Readme folder

### Pipeline:
The pipeline for detecting lane lines on the road is as follows:

1. Reading the input images/videos from the test_input folder.
2. Apply grayscale transform to convert RGB into grayscale.
3. Reducing noise using Gaussian noise kernel.
4. Apply Canny transform to get edges.
5. Mask the image and apply region of interest.
6. Apply Hough transform to find lines and drawing lines on the image with the desired color and thickness.
7. Blending the Hough output image and the unprocessed image.
8. Plotting and saving the output images/videos into the test_output folder.

Main stages of lane line detection can be seen below: 
<img src="readme_images/SDC_LaneLines_Stages.png" width="7500" alt="Six stages of lane line detection">
