import cv2
import os
import numpy as np
from collections import deque, Counter

# ------------------------------ CONFIG ------------------------------ #
USE_WEBCAM = True  # Set to True to use webcam
# VIDEO_PATH = r"C:\Users\YUKTA.PANCHAL\PycharmProjects\face_gender_detection\video_data\5.mp4"
# --------------------------------------------------------------------- #

# Load model paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Label definitions
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load pre-trained models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


# Face detection with multiple scales
def faceBox(net, frame, conf_threshold=0.5):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * frameWidth))
            y1 = max(0, int(detections[0, 0, i, 4] * frameHeight))
            x2 = min(frameWidth, int(detections[0, 0, i, 5] * frameWidth))
            y2 = min(frameHeight, int(detections[0, 0, i, 6] * frameHeight))

            # Filter out very small faces
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                bboxes.append([x1, y1, x2, y2])
    return bboxes


# Multiple preprocessing techniques for robust gender detection
def preprocess_face_multiple(face):
    preprocessed_faces = []

    # Original face with basic preprocessing
    face_orig = cv2.resize(face, (227, 227))
    face_orig = cv2.GaussianBlur(face_orig, (3, 3), 0)
    preprocessed_faces.append(face_orig)

    # Histogram equalized version
    face_eq = cv2.resize(face, (227, 227))
    face_eq_gray = cv2.cvtColor(face_eq, cv2.COLOR_BGR2GRAY)
    face_eq_gray = cv2.equalizeHist(face_eq_gray)
    face_eq = cv2.cvtColor(face_eq_gray, cv2.COLOR_GRAY2BGR)
    preprocessed_faces.append(face_eq)

    # CLAHE enhanced version
    face_clahe = cv2.resize(face, (227, 227))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_clahe_gray = cv2.cvtColor(face_clahe, cv2.COLOR_BGR2GRAY)
    face_clahe_gray = clahe.apply(face_clahe_gray)
    face_clahe = cv2.cvtColor(face_clahe_gray, cv2.COLOR_GRAY2BGR)
    preprocessed_faces.append(face_clahe)

    # Brightness adjusted version
    face_bright = cv2.resize(face, (227, 227))
    face_bright = cv2.convertScaleAbs(face_bright, alpha=1.2, beta=20)
    preprocessed_faces.append(face_bright)

    return preprocessed_faces


# Ensemble gender prediction with multiple models and preprocessing
def predict_gender_ensemble(face, genderNet):
    preprocessed_faces = preprocess_face_multiple(face)

    gender_votes = []
    confidence_scores = []

    for processed_face in preprocessed_faces:
        # Create blob
        blob = cv2.dnn.blobFromImage(processed_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()

        male_conf = genderPred[0][0]
        female_conf = genderPred[0][1]

        # Apply dynamic bias correction based on confidence difference
        conf_diff = abs(male_conf - female_conf)

        if conf_diff < 0.3:  # Low confidence, apply stronger bias correction
            female_conf *= 1.25
        elif conf_diff < 0.6:  # Medium confidence, apply moderate bias correction
            female_conf *= 1.15
        else:  # High confidence, apply minimal bias correction
            female_conf *= 1.05

        # Determine gender
        if female_conf > male_conf:
            gender_votes.append('Female')
            confidence_scores.append(female_conf)
        else:
            gender_votes.append('Male')
            confidence_scores.append(male_conf)

    # Weighted voting based on confidence scores
    weighted_votes = {'Male': 0, 'Female': 0}
    for vote, conf in zip(gender_votes, confidence_scores):
        weighted_votes[vote] += conf

    # Determine final gender
    final_gender = max(weighted_votes, key=weighted_votes.get)
    confidence_ratio = weighted_votes[final_gender] / sum(weighted_votes.values())

    return final_gender, confidence_ratio, weighted_votes


# Enhanced age prediction
def predict_age_enhanced(face, ageNet):
    # Use best preprocessing for age detection
    face_processed = cv2.resize(face, (227, 227))
    face_processed = cv2.GaussianBlur(face_processed, (3, 3), 0)

    blob = cv2.dnn.blobFromImage(face_processed, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    ageNet.setInput(blob)
    agePred = ageNet.forward()

    return ageList[agePred[0].argmax()]


# Advanced smoothing with confidence-based weighting
class AdvancedSmoother:
    def __init__(self, max_history=20):
        self.gender_history = deque(maxlen=max_history)
        self.gender_confidence = deque(maxlen=max_history)
        self.age_history = deque(maxlen=max_history)
        self.stable_gender = None
        self.stable_counter = 0

    def add_prediction(self, gender, confidence, age):
        self.gender_history.append(gender)
        self.gender_confidence.append(confidence)
        self.age_history.append(age)

    def get_stable_prediction(self):
        if len(self.gender_history) < 3:
            return self.gender_history[-1] if self.gender_history else "Unknown", \
                   self.age_history[-1] if self.age_history else "Unknown"

        # Confidence-weighted gender prediction
        recent_predictions = list(self.gender_history)[-10:]  # Last 10 predictions
        recent_confidences = list(self.gender_confidence)[-10:]

        weighted_votes = {'Male': 0, 'Female': 0}
        for pred, conf in zip(recent_predictions, recent_confidences):
            weighted_votes[pred] += conf

        # Require strong evidence for gender change
        total_weight = sum(weighted_votes.values())
        if total_weight > 0:
            female_ratio = weighted_votes['Female'] / total_weight
            male_ratio = weighted_votes['Male'] / total_weight

            if female_ratio >= 0.65:
                predicted_gender = 'Female'
            elif male_ratio >= 0.65:
                predicted_gender = 'Male'
            else:
                # If uncertain, stick with previous stable prediction
                predicted_gender = self.stable_gender if self.stable_gender else \
                    max(weighted_votes, key=weighted_votes.get)
        else:
            predicted_gender = "Unknown"

        # Update stable prediction
        if predicted_gender == self.stable_gender:
            self.stable_counter += 1
        else:
            self.stable_counter = 1
            self.stable_gender = predicted_gender

        # Age prediction with median smoothing
        if len(self.age_history) >= 5:
            age_indices = [ageList.index(age) for age in list(self.age_history)[-7:]]
            median_age_index = sorted(age_indices)[len(age_indices) // 2]
            smoothed_age = ageList[median_age_index]
        else:
            smoothed_age = self.age_history[-1]

        return predicted_gender, smoothed_age


# Open video or webcam
video = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)

if not video.isOpened():
    print("Error: Could not open video or webcam.")
    exit()

# Initialize advanced smoother
smoother = AdvancedSmoother()

# Process each frame
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        print("End of stream or failed to grab frame.")
        break

    frame_count += 1

    # Flip frame for webcam (mirror effect)
    if USE_WEBCAM:
        frame = cv2.flip(frame, 1)

    bboxes = faceBox(faceNet, frame)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Enhanced ensemble gender prediction
        gender, confidence_ratio, vote_details = predict_gender_ensemble(face, genderNet)

        # Enhanced age prediction
        age = predict_age_enhanced(face, ageNet)

        # Add to smoother
        smoother.add_prediction(gender, confidence_ratio, age)

        # Get stable prediction
        stable_gender, stable_age = smoother.get_stable_prediction()

        # Create display label
        main_label = f"{stable_gender}, {stable_age}"
        confidence_text = f"Conf: {confidence_ratio:.2f}"
        detail_text = f"M:{vote_details['Male']:.2f} F:{vote_details['Female']:.2f}"

        # Color coding based on confidence
        if confidence_ratio > 0.8:
            color = (0, 255, 0)  # Green - High confidence
        elif confidence_ratio > 0.6:
            color = (0, 255, 255)  # Yellow - Medium confidence
        else:
            color = (0, 0, 255)  # Red - Low confidence

        # Display results
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, main_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, confidence_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, detail_text, (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Display frame info
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Robust Age-Gender Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()