### **1. Importing Libraries**

```python
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from google.cloud import firestore
from google.oauth2 import service_account
```

- **`cv2`**: OpenCV library for computer vision tasks, used for video capturing and object detection.
- **`numpy`**: Numerical computations, used for array manipulations.
- **`scipy.spatial.distance`**: Computes distances between objects; here, it's used for centroid distance calculations.
- **`collections.OrderedDict`**: Maintains the order of keys; used to track object IDs and their locations.
- **`google.cloud.firestore` and `google.oauth2.service_account`**: Enable Firestore database connectivity for storing real-time data about detected people.

---

### **2. Firestore Initialization**

```python
credentials = service_account.Credentials.from_service_account_file("/home/temoc/Desktop/plant-automation/CREDS.json")
db = firestore.Client(credentials=credentials)
```

- **Firestore Initialization**: Authenticates Firestore access using credentials from the provided JSON file.
- **Firestore Client**: Allows interaction with Firestore to read/write data.

---

### **3. Centroid Tracker Class**

```python
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
```

- Tracks detected objects (e.g., people) across frames using centroids (object centers).
- **Attributes**:
  - `nextObjectID`: Assigns unique IDs to new objects.
  - `objects`: Dictionary mapping object IDs to centroids.
  - `disappeared`: Tracks how many frames an object has been missing.
  - `maxDisappeared`: Threshold to remove objects after being unseen for a given number of frames.

```python
    def register(self, centroid):
        ...
```

- Adds a new object with a centroid and assigns it an ID.

```python
    def deregister(self, objectID):
        ...
```

- Removes an object that has been unseen for too long.

```python
    def update(self, rects):
        ...
```

- Updates tracked objects based on detected bounding boxes (`rects`):
  1. Computes centroids from bounding boxes.
  2. Matches new centroids to existing objects using distance calculations.
  3. Updates positions or removes objects based on visibility.

---

### **4. Loading the MobileNet SSD Model**

```python
prototxt = '/home/temoc/Desktop/plant-automation/deploy.prototxt'
model = '/home/temoc/Desktop/plant-automation/mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)
```

- **MobileNet SSD**: A pre-trained deep learning model for object detection.
- **Prototxt**: Configuration file defining the model structure.
- **Caffemodel**: Weights for the pre-trained model.
- **`readNetFromCaffe`**: Loads the model for use.

---

### **5. Initializing the Webcam**

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
```

- Captures video from the default webcam (`0`).
- Exits with an error message if the webcam fails to initialize.

---

### **6. Variables and Centroid Tracker Initialization**

```python
tracker = CentroidTracker(maxDisappeared=50)
total_foot_traffic = 0
previous_people_count = 0
```

- **Tracker**: Tracks detected people across frames.
- **`total_foot_traffic`**: Cumulative count of people entering the frame.
- **`previous_people_count`**: Tracks the number of people from the previous frame to calculate changes.

---

### **7. Main Video Processing Loop**

#### Reading Frames

```python
ret, frame = cap.read()
if not ret:
    break
frame_resized = cv2.resize(frame, (300, 300))
```

- Reads the current frame from the webcam.
- Resizes the frame to 300x300 for faster processing.

---

#### Object Detection

```python
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()
```

- Converts the frame into a format suitable for the deep learning model (`blobFromImage`).
- Passes the blob through the model to get detections.

---

#### Processing Detections

```python
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.4:
        ...
```

- Iterates over all detected objects and filters them based on confidence (`> 0.4`).

```python
if idx == 15:
    box = detections[0, 0, i, 3:7] * ...
    rects.append((x1, y1, x2, y2))
```

- Checks if the detected object is a person (`class ID 15` for MobileNet SSD).
- Extracts the bounding box coordinates and appends them to `rects`.

---

#### Centroid Tracking

```python
objects = tracker.update(rects)
current_people_count = len(tracker.objects)
```

- Updates the tracker with new bounding boxes and computes the current number of people.

```python
if current_people_count > previous_people_count:
    total_foot_traffic += current_people_count - previous_people_count
```

- Updates the cumulative foot traffic when new people enter the frame.

---

#### Firebase Updates

```python
db.collection('room_tracking').document('current_status').set({
    'current_people': current_people_count,
    'timestamp': firestore.SERVER_TIMESTAMP
})
db.collection('room_tracking').document('daily_traffic').set({
    'total_foot_traffic': total_foot_traffic,
    'timestamp': firestore.SERVER_TIMESTAMP
})
```

- Sends real-time data to Firestore:
  - **`current_status`**: Tracks the number of people currently in the frame.
  - **`daily_traffic`**: Tracks the cumulative number of people detected.

---

#### Visual Feedback

```python
cv2.putText(frame_resized, f'Current People: {current_people_count}', ...)
cv2.putText(frame_resized, f'Foot Traffic: {total_foot_traffic}', ...)
cv2.imshow('People Detection and Tracking', frame_resized)
```

- Displays the count of people in the frame and cumulative foot traffic on the video feed.

---

#### Exit Condition

```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

- Exits the loop when the user presses the "q" key.

---

### **8. Cleanup**

```python
cap.release()
cv2.destroyAllWindows()
```

- Releases the webcam and closes all OpenCV windows.
