import cv2
import time

def record_video(output_file, duration_sec=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 640))

    start_time = time.time()
    elapsed_time = 0

    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)

    while elapsed_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

        elapsed_time = time.time() - start_time

    cap.release()
    out.release()

output_file = 'output.mp4'
record_video(output_file, duration_sec=30)
