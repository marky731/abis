import cv2
import numpy as np
import time
from image import *
from functions import *

url = "http://192.168.1.103:8080/video" # Your url here
# url = "http://10.151.65.233:8080/video"
cap = cv2.VideoCapture(url)

pTime = 0
cTime = 0

my_images = []
N_SLICES = 5

def is_line_contour(contour):
    # Approximate the contour with a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 2 vertices (indicating a line)
    return len(approx) == 2

for i in range(N_SLICES):
    my_images.append(Image())

while True:

    ret, frame = cap.read()

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

    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Convert to Gray Scale
    
    SlicePart(img, my_images, N_SLICES)
    img = RepackImages(my_images)

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
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    max_contour = None

    # Threshold the image
    ret, threshold_img = cv2.threshold(mask_color, 127, 255, cv2.THRESH_BINARY)
    # Find contours in the threshold image
    contours, _ = cv2.findContours(threshold_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on the original image
    contours = [contour for contour in contours if cv2.matchShapes(max_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0) > 0.01]
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    try:
        if contours:
            max_contour = max(contours, key = cv2.contourArea)
        # cv2.drawContours(frame, main_contour, -1, (255, 255, 255), 4)
    except Exception as e:
        print("exeption :", e)

    if max_contour is not None and center_contour is not None:
        if max_contour.all() == center_contour.all():
            print("center_contour == max_contour")
        else:
            print("not")
        # max_contour = max_contour.astype(np.float32)
        if is_line_contour(max_contour):
            print("--line--")
        

    # printing fps, not necessary
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("frame_1", img)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break