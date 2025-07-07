import cv2
from deepface import DeepFace

# Initialize webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    try:
        # Convert BGR (OpenCV) to RGB (DeepFace needs this)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the image
        results = DeepFace.analyze(rgb_frame, actions=['age', 'gender'], enforce_detection=True)

        # Handle single face or multiple faces
        if not isinstance(results, list):
            results = [results]

        for result in results:
            age = result.get('age')
            gender = result.get('dominant_gender')
            region = result.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display age and gender
            label = f"{gender.capitalize()}, {int(age)} yrs" if age is not None else gender
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

    except Exception as e:
        print("Face not detected:", e)

    # Show frame
    cv2.imshow("Age-Gender Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
