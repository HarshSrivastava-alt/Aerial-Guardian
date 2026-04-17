import cv2
from collections import deque

track_history = {}

def draw_tracks(frame, tracks):
    for tid, box in tracks.items():
        x1, y1, x2, y2 = map(int, box)
        center = ((x1+x2)//2, (y1+y2)//2)

        if tid not in track_history:
            track_history[tid] = deque(maxlen=30)

        track_history[tid].append(center)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        for i in range(1, len(track_history[tid])):
            cv2.line(frame,
                     track_history[tid][i-1],
                     track_history[tid][i],
                     (255,0,0), 2)

    return frame
