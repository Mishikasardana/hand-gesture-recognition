# hand-gesture-recognition
README - Hand Gesture Recognition using OpenCV

Project Overview:
-----------------
This project uses computer vision techniques to detect and classify hand gestures (ONE to FIVE fingers) in real-time using a webcam. 
The algorithm processes video frames, detects skin regions, finds contours, computes convexity defects, and counts fingers to identify gestures.

Source Code:
------------
- elc.py : Main Python script that captures video from the webcam, performs gesture detection, and displays results.
- Uses OpenCV for image processing and scikit-learn for evaluation metrics.

Datasets:
---------
- No external datasets are required.
- The system works on live video input from the webcam.
- The detection is based on color segmentation (skin color range in HSV) and contour analysis.

Dependencies:
-------------
- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn

You can install required packages using:
    pip install opencv-python numpy scikit-learn

How to Run:
-----------
1. Make sure you have a working webcam connected.
2. Install dependencies listed above.
3. Run the script using:
    python elc.py
4. A window will open displaying the webcam feed with the hand gesture detected and labeled.
5. Press 'q' to quit the application.
6. After quitting, evaluation metrics like Accuracy, Precision, Recall, F1-score, and Mean Average Precision (mAP) will be printed in the console based on the detected gestures during the session.

Notes:
------
- The gesture detection accuracy depends on lighting and background conditions.
- The skin color range in HSV is pre-defined but can be adjusted for different environments.
- For better evaluation results, perform gestures clearly in front of the camera during runtime.


