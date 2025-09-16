import cv2
import multiprocessing as mp
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

MODEL_PATH = "pesos/best.pt"

# --- Função de análise do pack ---
def analyze_pack_alignment(frame, model):
    """
    Analyze pack alignment (rows x cols) using YOLOv8 nano.
    Returns status string.
    """
    results = model(frame, verbose=False)
    centers, box_heights = [], []

    for res in results:
        for box in res.boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if box.conf[0] > 0.5:
                centers.append(((x1 + x2)//2, (y1 + y2)//2))
                box_heights.append(y2 - y1)

    status = "Not aligned"

    if len(centers) >= 4:
        pts = np.array(centers, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        rot_matrix = cv2.getRotationMatrix2D(tuple(rect[0]), angle, 1.0)
        rotated = cv2.transform(np.array([pts]), rot_matrix)[0]
        rotated = rotated[np.argsort(rotated[:,1])]  # sort by Y

        tol_y = np.mean(box_heights)*0.5 if box_heights else 20
        rows, current_row, ref_y = [], [], rotated[0][1]

        for p in rotated:
            if abs(p[1]-ref_y) < tol_y:
                current_row.append(p)
            else:
                rows.append(current_row)
                current_row = [p]
                ref_y = p[1]
        rows.append(current_row)

        counts = [len(r) for r in rows]
        max_cols, n_rows = max(counts), len(rows)
        expected, detected = max_cols*n_rows, len(centers)

        if detected < expected:
            status = f"Incomplete pack {max_cols}x{n_rows}: {detected}/{expected}"
        elif len(set(counts)) == 1 and detected == expected:
            status = f"Aligned: {max_cols}x{n_rows} ({expected} bottles)"
        else:
            status = f"Not aligned, row counts: {counts}"

    return status

# --- Processo de captura ---
def capture_frames(queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if ret:
            if queue.full():
                try:
                    queue.get_nowait()  # descarta frame antigo
                except:
                    pass
            queue.put(frame)
    cap.release()

# --- Processo de inferência ---
def process_frames(queue):
    model = YOLO(MODEL_PATH)  # carrega modelo no processo
    while True:
        if not queue.empty():
            frame = queue.get()
            status = analyze_pack_alignment(frame, model)
            print(status)
            cv2.imshow("Pack Alignment Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# --- Main ---
if __name__ == "__main__":
    mp.set_start_method('spawn')  # importante para Raspberry Pi

    frame_queue = mp.Queue(maxsize=2)  # armazena apenas últimos frames

    p_capture = mp.Process(target=capture_frames, args=(frame_queue,))
    p_process = mp.Process(target=process_frames, args=(frame_queue,))

    p_capture.start()
    p_process.start()

    p_process.join()
    p_capture.terminate()
