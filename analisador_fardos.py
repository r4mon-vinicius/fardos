import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def analyze_pack_alignment(frame, model, expected_count=None):
    """
    Analyze a pack alignment (rows x cols) using YOLOv8 and grid analysis.
    """
    result_frame = frame.copy()
    results = model(frame, verbose=False)

    centers, box_heights = [], []
    for result in results:
        for box in result.boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > 0.5:
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                box_heights.append(y2 - y1)

    status, color = "Not aligned", (0, 0, 255)

    if len(centers) >= 4:
        pts = np.array(centers, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        rot_matrix = cv2.getRotationMatrix2D(tuple(rect[0]), angle, 1.0)
        rotated = cv2.transform(np.array([pts]), rot_matrix)[0]
        rotated = sorted(rotated, key=lambda p: (p[1], p[0]))

        tol_y = np.mean(box_heights) * 0.5 if box_heights else 20
        rows, row, ref_y = defaultdict(list), [], rotated[0][1]

        for p in rotated:
            if abs(p[1] - ref_y) < tol_y:
                row.append(p)
            else:
                rows[ref_y] = sorted(row, key=lambda x: x[0])
                row, ref_y = [p], p[1]
        rows[ref_y] = sorted(row, key=lambda x: x[0])

        counts = [len(r) for r in rows.values()]
        max_cols, n_rows = max(counts), len(rows)
        expected, detected = max_cols * n_rows, len(centers)

        if detected < expected:
            status = f"Incomplete pack {max_cols}x{n_rows}: {detected}/{expected}"
            color = (0, 165, 255)
        elif len(set(counts)) == 1 and detected == expected:
            status = f"Aligned: {max_cols}x{n_rows} ({expected} bottles)"
            color = (0, 255, 0)
        else:
            status = f"Not aligned, row counts: {counts}"
            color = (0, 0, 255)

    for c in centers:
        cv2.circle(result_frame, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)

    cv2.putText(result_frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    return result_frame, status


if __name__ == "__main__":
    MODEL_PATH = "pesos/best.pt"
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        result_frame, status = analyze_pack_alignment(frame, model)
        cv2.imshow("Pack Alignment Analysis", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
