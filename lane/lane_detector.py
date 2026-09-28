import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        pass

    def detect(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width, int(height * 0.6)),
            (0, int(height * 0.6))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, maxLineGap=150)
        line_img = np.zeros_like(frame)

        if lines is not None:
            left_lines, right_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue  # vertical line, no slope
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    continue  # too flat to be a lane line
                intercept = y1 - slope * x1
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))

            def average_line(group):
                if not group:
                    return None
                slope, intercept = np.mean(group, axis=0)
                y1, y2 = height, int(height * 0.6)
                xa = (y1 - intercept) / slope
                xb = (y2 - intercept) / slope
                if not (np.isfinite(xa) and np.isfinite(xb)):
                    return None
                xa = int(np.clip(xa, -width, 2 * width))
                xb = int(np.clip(xb, -width, 2 * width))
                return (xa, int(y1), xb, int(y2))

            for ln in (average_line(left_lines), average_line(right_lines)):
                if ln is not None:
                    x1, y1, x2, y2 = ln
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 8)

        return cv2.addWeighted(frame, 0.8, line_img, 1, 1)