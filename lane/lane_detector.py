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
            (width, int(height*0.6)),
            (0, int(height*0.6))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, maxLineGap=150)
        line_img = np.zeros_like(frame)

        if lines is not None:
            left_lines, right_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope, intercept = parameters
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))

            def average_lines(lines):
                if len(lines) == 0:
                    return None
                slope, intercept = np.mean(lines, axis=0)
                y1, y2 = height, int(height * 0.6)
                x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
                return np.array([x1, y1, x2, y2])

            left = average_lines(left_lines)
            right = average_lines(right_lines)

            for line in [left, right]:
                if line is not None:
                    x1, y1, x2, y2 = line
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 8)

        combined = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
        return combined
