import cv2
import math
import argparse
import os # Import the os library to handle file paths

def highlightFace(net, frame, conf_threshold=0.8):
    """
    Detects faces in an image frame and returns the frame with bounding boxes
    and a list of the face coordinates.
    """
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition.')
parser.add_argument('--image', help='Path to input image file.')
args = parser.parse_args()

# --- Model and Configuration File Paths (Robust Path Handling) ---
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script's directory (the project root)
project_root = os.path.dirname(script_dir)
# Define the base path for the models relative to the project root
models_path = os.path.join(project_root, "models")

faceProto = os.path.join(models_path, "opencv_face_detector.pbtxt")
faceModel = os.path.join(models_path, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(models_path, "age_deploy.prototxt")
ageModel = os.path.join(models_path, "age_net.caffemodel")
genderProto = os.path.join(models_path, "gender_deploy.prototxt")
genderModel = os.path.join(models_path, "gender_net.caffemodel")

# --- Model Preprocessing Values ---
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# --- Class Labels ---
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
# --- Load the pre-trained networks ---
print("Loading networks...")
try:
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    print("Networks loaded successfully.")
except cv2.error as e:
    print(f"Error loading model files: {e}")
    print(f"Please ensure the model files are in the '{models_path}' directory.")
    exit()

# --- Open the video stream or image file ---
video_source = args.image if args.image else 0
video = cv2.VideoCapture(video_source)

if not video.isOpened():
    print(f"Cannot open video source: {video_source}")
    exit()

padding = 20

# --- Main Loop to Process Frames ---
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        if args.image:
            print("Processing complete. Press any key to exit.")
            cv2.waitKey(0)
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):
                     min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        label = f'{gender}, {age}'
        print(f"Detected: {label}")
        cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender Detection", resultImg)
    if args.image:
        continue

# --- Cleanup ---
video.release()
cv2.destroyAllWindows()
