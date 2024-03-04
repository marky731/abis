import cv2
import time
import threading
from ultralytics import YOLO
import math
import numpy as np
from image import *
from functions import *

class Camera:
    def __init__(self, model_file, video_port):
        self.model_file = model_file
        self.video_port = video_port
        self.active = True
        self.isCameraActive = False
        self.camera = cv2.VideoCapture(self.video_port)
        self.Frame = None
        self.cameraThread = threading.Thread(target=self.getFrame)

        self.model = YOLO(self.model_file)
        self.cameraThread.start()

        while not self.isCameraActive:
            self.isCameraActive = self.camera.isOpened()

    def openCamera(self):
        return cv2.VideoCapture(self.video_port)

    def getFrame(self):
        while self.active:
            ret, frame = self.camera.read()
            if ret:
                self.Frame = frame
                self.isCameraActive = ret
            else:
                print("getFrame() error")

        self.camera.release()

    def return_frame(self):
        return self.Frame

    def detectObject(self, frame):
        results = self.model(frame, stream=True)
        return results

shape_labels = ['circle', 'hexagon', 'rectanle', 'square', 'triangle']
captured_frames = {shape: 0 for shape in shape_labels}
detected_frames = {shape: 0 for shape in shape_labels}
max_frames_per_shape = 5  # Set the desired number of frames per shape

my_images = []
N_SLICES = 4 # how many slice do you want to divide the line which is gonna be detected

for i in range(N_SLICES):
    my_images.append(Image())

url = "http://192.168.1.104:8080/video" # Your url here
"video.mov"
the_camera = Camera('best_2.pt', url)

pTime = 0
cTime = 0 # to calculate fps 

def is_line_contour(contour):
    # Approximate the contour with a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 2 vertices (indicating a line)
    return len(approx) == 2

while True:
    # print("111")
    # the_camera.getFrame()
    # print("camera is active: ", the_camera.isCameraActive)
    if the_camera.isCameraActive:
        direction = 0
        try:
            frame = the_camera.return_frame()
            height, width = frame.shape[:2]

            centerX, centerY = width // 2, height // 2 # will be used to determine the coordinates of objects

            position = the_camera.detectObject(frame)

            for p in position:

                boxes_ = p.boxes
                
                for box in boxes_:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    if confidence >= 0.7:
                        class_index = int(box.cls)
                        shape_label = shape_labels[class_index]

                        detected_frames[shape_label] += 1
                        print(shape_label, " : ", detected_frames[shape_label])


                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"{shape_label} {confidence}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 255, 255), 2)
                        
                        # capturing image
                        if (captured_frames[shape_label] < max_frames_per_shape) and (detected_frames[shape_label] > 2):
                            # Capture the frame only if it hasn't reached the limit for this shape
                            captured_frames[shape_label] += 1

                            # Save the captured frame to an image file
                            filename = f"captured_{shape_label}_{captured_frames[shape_label]}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"Captured frame {captured_frames[shape_label]} for {shape_label}.")

                        x_center_box = (x1 + x2) / 2
                        y_center_box = (y1 + y2) / 2

                        cor_x = x_center_box - centerX
                        cor_y = centerY - y_center_box
                        text = f"X: {cor_x}, Y: {cor_y}"

                        cv2.putText(frame, text, (x1, y2 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
            
            ### --------------beginning of line detection-------------- ###

            # Define the lower and upper bounds for white color
            lower_white = np.array([110,110,110])
            upper_white = np.array([200, 200, 200])
            lower_red = np.array([0, 0, 140])
            upper_red = np.array([100, 100, 255])

            mask_color = cv2.inRange(frame, lower_red, upper_red)
            img = frame.copy()
            img = cv2.bitwise_and(img, img, mask = mask_color)
            img = cv2.bitwise_not(img, img, mask = mask_color)
            img = (255-img)  # background of img is removed 

            SlicePart(img, my_images, N_SLICES)
            img = RepackImages(my_images) #line is divided into N_SLICES parts and x coordinate of each part is calculated

            for i in range(N_SLICES):
                direction += my_images[i].dir

            contour_color, _ = cv2.findContours(mask_color.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center_image = np.array(img.shape[1::-1]) // 2    
            center_contour = None
            min_distance = float('inf')
            for contour in contour_color:
                # Calculate the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Calculate the Euclidean distance to the center of the image
                    distance = np.linalg.norm(center_image - np.array([cX, cY]))
                    # Check if this contour is closer than the previous minimum
                    if distance < min_distance:
                        min_distance = distance
                        center_contour = contour
            # If a contour is found, draw a bounding box around it
            if center_contour is not None:
                x, y, w, h = cv2.boundingRect(center_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            max_contour = None

            # Threshold the image
            ret, threshold_img = cv2.threshold(mask_color, 127, 255, cv2.THRESH_BINARY)
            # Find contours in the threshold image
            contours, _ = cv2.findContours(threshold_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Draw the contours on the original image
            contours = [contour for contour in contours if cv2.matchShapes(max_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0) > 0.01]
            # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

            try:
                if contours:
                    max_contour = max(contours, key = cv2.contourArea)
                # cv2.drawContours(frame, main_contour, -1, (255, 255, 255), 4)
            except Exception as e:
                print("exeption(max contour) :", e)

            if max_contour is not None and center_contour is not None:
                if max_contour.all() == center_contour.all():
                    print("center_contour == max_contour")
                else:
                    print("center_contour doesn't equal to max_contour")
                # max_contour = max_contour.astype(np.float32)
                if is_line_contour(max_contour): # this function is usefull only when the line is thin, long and not very curvy
                    print("--line--")

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.6, (255, 0, 255), 3)
            cv2.putText(img, str(int(direction)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.6, (255, 0, 255), 3)
            cv2.imshow("img", img)
            cv2.imshow("frame", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                the_camera.active = False
                break

        except Exception as ee:
            print('Error: ', ee)