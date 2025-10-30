import cv2
import time
import numpy as np
from detection.yolo_detector import ObjectDetector
from detection.tracker import Tracker
from lane.lane_detector import LaneDetector
from utils.visualization import draw_detections


def main():
    # --- Initialize camera ---
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam or replace with video path
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- Initialize modules ---
    detector = ObjectDetector('yolov8s.pt')
    tracker = Tracker()
    lane_detector = LaneDetector()

    # --- Performance metrics ---
    start_time = time.time()
    frame_count = 0

    print("[INFO] Starting real-time perception system. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame not captured. Retrying...")
            continue  # skip dropped frames without stopping

        frame_count += 1
        t1 = time.time()

        # --- Object detection ---
        detections = detector.detect(frame)

        # --- Object tracking ---
        tracks = tracker.update(detections)

        # --- Lane detection ---
        frame_with_lanes = lane_detector.detect(frame)

        # --- Visualization ---
        output = draw_detections(frame_with_lanes, detections)

        # --- Compute FPS ---
        fps = 1.0 / (time.time() - t1 + 1e-8)
        elapsed = time.time() - start_time

        # --- Display FPS and elapsed time ---
        cv2.putText(output, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(output, f"Time: {elapsed:.1f}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # --- Display Output ---
        cv2.imshow('Perception System', output)

        # --- Key Controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting system.")
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"[INFO] Total runtime: {total_time:.2f}s, Frames processed: {frame_count}")


if __name__ == "__main__":
    main()



