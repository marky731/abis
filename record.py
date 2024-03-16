import cv2
import time

def record_video(output_file, duration_sec=10):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 640))

    start_time = time.time()
    elapsed_time = 0

    cap = cv2.VideoCapture(0)  # Assuming camera index 0, change if necessary
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)

    while elapsed_time < duration_sec:
        ret, frame = cap.read()
        if ret:
           
            frame = cv2.resize(frame, (640, 640))

            # Write the frame to the output video file
            out.write(frame)

            # Display the frame
            cv2.imshow('Recording', frame)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Display elapsed time (optional)
            print(f"Elapsed Time: {elapsed_time:.2f} sec")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_file = '/Users/mac/Desktop/Programming/Python/AbisYazılım/record.mp4' #dosya yolu degisecek
    record_video(output_file, duration_sec=6)
