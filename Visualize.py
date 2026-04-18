import cv2
import numpy as np

# Each tracked ID gets a unique color - we'll generate them randomly but consistently
def get_color(track_id):
    """
    Returns a unique BGR color for each track ID.
    Same ID always gets same color across frames.
    """
    np.random.seed(track_id * 10)
    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
    return color


def draw_tracks(frame, tracks, tail_points):
    """
    Draws bounding boxes, ID labels, and trajectory tails on the frame.
    
    tracks: list of [x1, y1, x2, y2, track_id]
    tail_points: dict of {track_id: [(cx, cy), ...]} storing last N positions
    """
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        color = get_color(track_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw ID label with background for readability
        label = f"ID:{track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Update tail points for this track
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if track_id not in tail_points:
            tail_points[track_id] = []
        tail_points[track_id].append((cx, cy))

        # Keep only last 30 positions
        if len(tail_points[track_id]) > 30:
            tail_points[track_id].pop(0)

        # Draw tail as connected line segments
        points = tail_points[track_id]
        for i in range(1, len(points)):
            # Tail fades out — older points are more transparent
            alpha = i / len(points)
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, points[i - 1], points[i], color, thickness)

    return frame, tail_points


def draw_info(frame, frame_num, fps, num_tracks):
    """
    Draws frame info overlay at the top of the frame.
    """
    info = f"Frame: {frame_num} | FPS: {fps:.1f} | Tracking: {num_tracks} persons"
    cv2.putText(frame, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
