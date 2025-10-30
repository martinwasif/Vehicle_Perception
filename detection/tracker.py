import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sort.sort import Sort

class Tracker:
    def __init__(self):
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

    def update(self, detections):
        det_array = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d["confidence"]
            det_array.append([x1, y1, x2, y2, conf])
        tracks = self.tracker.update(np.array(det_array))
        return tracks
