import cv2

def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["label"]} {d["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return frame
