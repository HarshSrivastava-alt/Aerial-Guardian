import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Only vehicle classes
vehicle_classes = [2, 3, 5, 7]

results = model.track(
    source="input_video.mp4",
    classes=vehicle_classes,
    persist=True
)

cap = cv2.VideoCapture("input_video.mp4")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 30,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=vehicle_classes, persist=True)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
