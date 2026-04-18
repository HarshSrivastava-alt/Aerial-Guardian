import cv2
import numpy as np
import os
import time
from pathlib import Path
from ultralytics import YOLO
from src.ego_motion import estimate_homography, warp_boxes
from src.visualize import draw_tracks, draw_info


# ── Dataset path ────────────────────────────────────────────────────────────
DATASET_PATH = Path("data/VisDrone2019-MOT-val/VisDrone2019-MOT-val/sequences")
OUTPUT_PATH  = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

# ── SAHI sliced inference (manual lightweight version) ───────────────────────
def slice_frame(frame, slice_size=320, overlap=0.2):
    """
    Divides the frame into overlapping patches.
    This helps detect small persons that YOLO might miss at full resolution.
    Returns list of (patch, x_offset, y_offset)
    """
    h, w = frame.shape[:2]
    step = int(slice_size * (1 - overlap))
    slices = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + slice_size, w)
            y2 = min(y + slice_size, h)
            patch = frame[y:y2, x:x2]
            slices.append((patch, x, y))

    return slices


def run_detection_on_slices(model, frame, conf=0.3, slice_size=320, overlap=0.2):
    """
    Runs YOLO detection on each slice and merges results back
    into full-frame coordinates. Only keeps 'person' class (class 0).
    """
    slices = slice_frame(frame, slice_size, overlap)
    all_boxes = []  # [x1, y1, x2, y2, confidence]

    for patch, ox, oy in slices:
        results = model(patch, verbose=False, conf=conf, classes=[0])

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_score = float(box.conf[0])

                # Translate patch coordinates back to full frame
                x1 += ox
                y1 += oy
                x2 += ox
                y2 += oy

                all_boxes.append([x1, y1, x2, y2, conf_score])

    # Apply Non-Maximum Suppression to remove duplicate detections
    if len(all_boxes) == 0:
        return []

    boxes_array = np.array([[b[0], b[1], b[2], b[3]] for b in all_boxes])
    scores      = np.array([b[4] for b in all_boxes])
    indices     = cv2.dnn.NMSBoxes(
        boxes_array.tolist(), scores.tolist(),
        score_threshold=0.1, nms_threshold=0.45
    )

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(all_boxes[i])

    return final_boxes


def process_sequence(sequence_path, model):
    """
    Processes one drone video sequence (folder of jpg frames).
    Returns average FPS for this sequence.
    """
    seq_name   = sequence_path.name
    frame_files = sorted(sequence_path.glob("*.jpg"))

    if len(frame_files) == 0:
        print(f"  No frames found in {seq_name}")
        return 0

    print(f"\n Processing: {seq_name} ({len(frame_files)} frames)")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    h, w        = first_frame.shape[:2]

    # Setup video writer for output
    out_path = OUTPUT_PATH / f"{seq_name}_output.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, 10, (w, h))

    # Initialize ByteTrack tracker
    tracker = model.predictor if hasattr(model, 'predictor') else None

    # State variables
    prev_gray    = None
    tail_points  = {}   # {track_id: [(cx,cy), ...]}
    frame_times  = []
    frame_num    = 0

    # We'll use ultralytics built-in tracker
    # Reset by re-initializing for each sequence
    track_history = {}

    for frame_path in frame_files:
        frame_start = time.time()
        frame       = cv2.imread(str(frame_path))

        if frame is None:
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Ego-motion compensation ──────────────────────────────────────
        H = np.eye(3)
        if prev_gray is not None:
            H = estimate_homography(prev_gray, curr_gray)

        # ── Detection using sliced inference ─────────────────────────────
        detections = run_detection_on_slices(model, frame, conf=0.1, slice_size=256)

        # Apply ego-motion correction to detected boxes
        if len(detections) > 0 and prev_gray is not None:
            raw_boxes    = [[d[0], d[1], d[2], d[3]] for d in detections]
            warped       = warp_boxes(raw_boxes, H)
            for i, d in enumerate(detections):
                detections[i][0] = warped[i][0]
                detections[i][1] = warped[i][1]
                detections[i][2] = warped[i][2]
                detections[i][3] = warped[i][3]

        # ── Tracking with ByteTrack ───────────────────────────────────────
        # Run ultralytics tracker on full frame for tracking
        track_results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.1,
            classes=[0],        # persons only
            tracker="bytetrack.yaml"
        )

        # Extract tracked boxes
        tracks = []
        if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
            boxes = track_results[0].boxes.xyxy.cpu().numpy()
            ids   = track_results[0].boxes.id.cpu().numpy()

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                tracks.append([x1, y1, x2, y2, tid])

        # ── Draw results ─────────────────────────────────────────────────
        elapsed  = time.time() - frame_start
        fps      = 1.0 / elapsed if elapsed > 0 else 0
        frame_times.append(elapsed)

        frame, tail_points = draw_tracks(frame, tracks, tail_points)
        frame = draw_info(frame, frame_num, fps, len(tracks))

        writer.write(frame)

        prev_gray = curr_gray
        frame_num += 1

        if frame_num % 50 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-50:])
            print(f"  Frame {frame_num}/{len(frame_files)} | Avg FPS: {avg_fps:.1f}")

    writer.release()
    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
    print(f"  Done! Avg FPS: {avg_fps:.1f} | Saved: {out_path}")
    return avg_fps


def main():
    print("=" * 50)
    print("  Aerial Guardian — Drone Person Tracker")
    print("=" * 50)

    # Load YOLOv8n model (downloads automatically ~6MB)
    print("\nLoading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded!")

    # Get all sequences
    sequences = sorted(DATASET_PATH.iterdir())
    sequences = [s for s in sequences if s.is_dir()][1:2]
    print(f"\nFound {len(sequences)} sequences to process")

    all_fps = []
    for seq_path in sequences:
        fps = process_sequence(seq_path, model)
        all_fps.append(fps)

    print("\n" + "=" * 50)
    print(f"  All sequences done!")
    print(f"  Overall Avg FPS: {np.mean(all_fps):.1f}")
    print(f"  Output videos saved in: outputs/")
    print("=" * 50)


if __name__ == "__main__":
    main()
