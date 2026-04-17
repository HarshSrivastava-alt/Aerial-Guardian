import cv2
import time
import numpy as np
from ultralytics import YOLO

# -------------------------------
# CONFIG
# -------------------------------
VIDEO_PATH = "input.mp4"   # replace with VisDrone video
OUTPUT_PATH = "output.mp4"
MODEL_PATH = "yolov8n.pt"

CONF_THRESHOLD = 0.3
IMG_SIZE = 960

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO(MODEL_PATH)

# -------------------------------
# BYTE TRACK (LIGHT VERSION)
# -------------------------------
from collections import deque

class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_id = 0
        self.tracks = {}
        self.lost = {}
        self.max_lost = max_lost

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        updated_tracks = {}
        used = set()

        for tid, tbox in self.tracks.items():
            best_iou = 0
            best_det = -1

            for i, det in enumerate(detections):
                if i in used:
                    continue
                iou_score = self.iou(tbox, det)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = i

            if best_iou > 0.3:
                updated_tracks[tid] = detections[best_det]
                used.add(best_det)
                self.lost[tid] = 0
            else:
                self.lost[tid] += 1
                if self.lost[tid] < self.max_lost:
                    updated_tracks[tid] = tbox

        # New tracks
        for i, det in enumerate(detections):
            if i not in used:
                updated_tracks[self.next_id] = det
                self.lost[self.next_id] = 0
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks


# -------------------------------
# DRAW TRAJECTORY
# -------------------------------
track_history = {}

def draw_tracks(frame, tracks):
    for tid, box in tracks.items():
        x1, y1, x2, y2 = map(int, box)
        center = ((x1+x2)//2, (y1+y2)//2)

        if tid not in track_history:
            track_history[tid] = deque(maxlen=30)

        track_history[tid].append(center)

        # Bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Trail
        for i in range(1, len(track_history[tid])):
            cv2.line(frame,
                     track_history[tid][i-1],
                     track_history[tid][i],
                     (255,0,0), 2)

    return frame


# -------------------------------
# VIDEO PROCESSING
# -------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_input,
    (width, height)
)

tracker = SimpleTracker()

frame_count = 0
total_time = 0

print("🚀 Processing started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # ---------------- DETECTION ----------------
    results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2])

    # ---------------- TRACKING ----------------
    tracks = tracker.update(detections)

    # ---------------- DRAW ----------------
    frame = draw_tracks(frame, tracks)

    end = time.time()
    frame_time = end - start
    total_time += frame_time
    frame_count += 1

    fps = 1 / frame_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    out.write(frame)
    cv2.imshow("Aerial Guardian", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# -------------------------------
# FINAL FPS REPORT
# -------------------------------
avg_fps = frame_count / total_time
print(f"\n✅ Average FPS: {avg_fps:.2f}")
print("🎥 Output saved to:", OUTPUT_PATH)
