Real-Time Age and Gender Detection
This project uses Python and OpenCV to detect the age and gender of individuals from a live webcam feed or a static image file. It leverages pre-trained deep learning models to perform real-time analysis.

Features
Real-Time Detection: Analyzes video from a webcam to detect faces, genders, and age ranges in real-time.

Image File Analysis: Can also process a single image file provided via the command line.

Structured Code: The project is organized into separate directories for source code, models, and test images for clarity and scalability.

Project Structure
├── src/
│   └── detect.py
├── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── ... (and other model files)
├── images/
│   └── ... (sample images for testing)
├── README.md
└── LICENSE

Prerequisites
Before you begin, ensure you have Python installed on your system. This project requires the following Python library:

OpenCV: A powerful library for computer vision tasks.

Installation
Clone or download the repository:
Get a local copy of this project on your machine.

Install the required library:
Open your terminal or command prompt and run the following command to install OpenCV:

pip install opencv-python

Usage
You can run the script in two modes: webcam analysis or single image analysis.

1. Webcam Analysis
To start the real-time detection using your webcam, navigate to the project's root directory in your terminal and run:

python src/detect.py

A window will appear showing your webcam feed with bounding boxes and labels for detected faces. Press the 'q' key to exit.

2. Image File Analysis
To analyze a single image, use the --image flag followed by the path to your image.

python src/detect.py --image images/man1.jpg

The script will process the image, display the results in a new window, and print the detected information to the console. Press any key to close the image window.

How It Works
The script uses pre-trained Caffe models for detection:

A Face Detection model first identifies the location of any faces in the frame.

For each detected face, the ROI (Region of Interest) is extracted.

This ROI is then passed to two separate models:

A Gender Prediction model classifies the face as 'Male' or 'Female'.

An Age Prediction model classifies the face into one of eight age ranges.

The results are then drawn onto the original frame and displayed.

License
This project is licensed under the MIT License. See the LICENSE file for details.