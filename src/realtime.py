import cv2
import numpy as np
import tensorflow as tf
import time

# =========================
# LOAD EMOTION MODEL
# =========================

model = tf.keras.models.load_model(
    "models/best_emotion_model.h5"
)

# =========================
# EMOTION LABELS
# =========================

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]

# =========================
# LOAD FACE DETECTOR
# =========================

prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(
    prototxt_path,
    model_path
)

# =========================
# START WEBCAM
# =========================

cap = cv2.VideoCapture(0)

# Camera optimization
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS calculation
prev_time = 0

print("\n===================================")
print(" REAL-TIME EMOTION DETECTION STARTED ")
print(" Press 'Q' to Quit ")
print("===================================\n")

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    h, w = frame.shape[:2]

    # =========================
    # FACE DETECTION
    # =========================

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    # =========================
    # PROCESS DETECTIONS
    # =========================

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Confidence threshold
        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (startX, startY, endX, endY) = box.astype("int")

            # Ensure coordinates inside frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # =========================
            # FACE ROI
            # =========================

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            # Convert to grayscale
            gray_face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

            # Resize
            gray_face = cv2.resize(
                gray_face,
                (48, 48)
            )

            # Normalize
            gray_face = gray_face / 255.0

            # Reshape for model
            gray_face = np.reshape(
                gray_face,
                (1, 48, 48, 1)
            )

            # =========================
            # EMOTION PREDICTION
            # =========================

            prediction = model.predict(
                gray_face,
                verbose=0
            )

            emotion_index = np.argmax(prediction)

            emotion = emotion_labels[emotion_index]

            confidence_score = np.max(prediction)

            # =========================
            # DRAW RESULTS
            # =========================

            text = f"{emotion} ({confidence_score:.2f})"

            # Rectangle
            cv2.rectangle(
                frame,
                (startX, startY),
                (endX, endY),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                frame,
                text,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # =========================
    # FPS DISPLAY
    # =========================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # =========================
    # SHOW FRAME
    # =========================

    cv2.imshow(
        "Real-Time Emotion Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE RESOURCES
# =========================

cap.release()
cv2.destroyAllWindows()