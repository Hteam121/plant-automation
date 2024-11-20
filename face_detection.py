import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from google.cloud import firestore
from google.oauth2 import service_account
import datetime

# Initialize Firestore
credentials = service_account.Credentials.from_service_account_file("/path/to/your/creds.json")
db = firestore.Client(credentials=credentials)

# Load the classifier and label encoder
classifier = load("/path/to/classifier.joblib")
label_encoder = load("/path/to/label_encoder.joblib")

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def update_firestore(name, timestamp):
    """Update Firestore with detected face and timestamp."""
    # Update the latest detection
    db.collection('face_detections').document(name).set({
        'last_detected': timestamp
    })

    # Add a new entry to the history
    db.collection('face_detections_history').add({
        'name': name,
        'timestamp': timestamp
    })

def recognize_and_update(frame):
    """Recognize faces and update Firestore."""
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face = frame[y:y + h, x:x + w]

            # Resize and normalize face
            face_resized = cv2.resize(face, (160, 160))
            face_flattened = face_resized.flatten() / 255.0

            # Predict with SVM
            embedding = np.expand_dims(face_flattened, axis=0)
            predictions = classifier.predict_proba(embedding)
            max_index = np.argmax(predictions)
            predicted_label = label_encoder.inverse_transform([max_index])[0]
            confidence = predictions[0][max_index]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Update Firestore
            timestamp = datetime.datetime.now().isoformat()
            update_firestore(predicted_label, timestamp)

    return frame

# Initialize Webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame for face recognition
    processed_frame = recognize_and_update(frame)

    # Show the video feed
    cv2.imshow("Face Recognition", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
