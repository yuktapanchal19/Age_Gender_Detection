import cv2
import os

def faceBox(net, frame, conf_threshold=0.7):
    frameDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frameWidth - 1, x2)
            y2 = min(frameHeight - 1, y2)

            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameDnn, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frameDnn, bboxes


# MODEL FILE PATHS
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# CONSTANTS
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# LOAD MODELS
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# VIDEO FILE PATH (make sure it's valid)
video_path = r'C:\Users\YUKTA.PANCHAL\PycharmProjects\face_gender_detection\video_data\4.mp4'
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

# MAIN LOOP
while True:
    ret, frame = video.read()
    if not ret:
        print("End of video or can't read frame.")
        break

    frameFace, bboxes = faceBox(faceNet, frame)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]

        # Skip if face area is too small or invalid
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.rectangle(frameFace, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
        cv2.putText(frameFace, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender", frameFace)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
