import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from google.cloud import firestore
from google.oauth2 import service_account

# Initialize Firestore
credentials = service_account.Credentials.from_service_account_file("/home/temoc/Desktop/plant-automation/CREDS.json")
db = firestore.Client(credentials=credentials)

# Centroid Tracker for tracking objects
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Load MobileNet SSD model for object detection
prototxt = '/home/temoc/Desktop/plant-automation/deploy.prototxt'
model = '/home/temoc/Desktop/plant-automation/mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize centroid tracker
tracker = CentroidTracker(maxDisappeared=50)

# Variables to count total entries
total_foot_traffic = 0
previous_people_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (300, 300))

    # Prepare input blob for object detection
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    rects = []

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Confidence threshold
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])

            # Check if detection is a person (class id 15 in MobileNet SSD)
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([frame_resized.shape[1], frame_resized.shape[0], frame_resized.shape[1], frame_resized.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")

                rects.append((x1, y1, x2, y2))

                # Draw bounding box
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update centroid tracker with bounding boxes
    objects = tracker.update(rects)

    # Update people count and foot traffic
    current_people_count = len(tracker.objects)

    if current_people_count > previous_people_count:
        total_foot_traffic += current_people_count - previous_people_count

    previous_people_count = current_people_count

    # Reset current people count to 0 if no detections
    if len(rects) == 0:
        current_people_count = 0

    # Update Firebase
    db.collection('room_tracking').document('current_status').set({
        'current_people': current_people_count,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    db.collection('room_tracking').document('daily_traffic').set({
        'total_foot_traffic': total_foot_traffic,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    # Display counts
    cv2.putText(frame_resized, f'Current People: {current_people_count}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0,0), 2)
    cv2.putText(frame_resized, f'Foot Traffic: {total_foot_traffic}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show the video feed
    cv2.imshow('People Detection and Tracking', frame_resized)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
