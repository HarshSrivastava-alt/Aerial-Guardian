class Tracker:
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

        for i, det in enumerate(detections):
            if i not in used:
                updated_tracks[self.next_id] = det
                self.lost[self.next_id] = 0
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks
