import cv2
import time
from detect import Detector
from tracker.tracker import Tracker
from utils.draw import draw_tracks
from config import VIDEO_PATH, OUTPUT_PATH

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(3))
height = int(cap.get(4))
fps_input = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_input,
    (width, height)
)

detector = Detector()
tracker = Tracker()

frame_count = 0
total_time = 0

print("🚀 Running pipeline...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    detections = detector.detect(frame)
    tracks = tracker.update(detections)

    frame = draw_tracks(frame, tracks)

    end = time.time()
    fps = 1 / (end - start)

    frame_count += 1
    total_time += (end - start)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    out.write(frame)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Avg FPS: {frame_count/total_time:.2f}")
