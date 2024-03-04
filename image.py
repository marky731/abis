import numpy as np
import cv2

class Image:
    
    def __init__(self):
        self.image = None
        self.contourCenterX = 0
        self.MainContour = None
        self.dir = 0
        
    def Process(self):
        imgray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) #Convert to Gray Scale
        ret, thresh = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY_INV) #Get Threshold

        self.contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Get contour from threshold of self.image
        
        self.prev_MC = self.MainContour # previous main contuour
        if self.contours:
            self.MainContour = max(self.contours, key=cv2.contourArea) # choose the biggest contour as main contour
        
            self.height, self.width  = self.image.shape[:2]

            self.middleX = int(self.width/2) #Get X coordenate of the middle point
            self.middleY = int(self.height/2) #Get Y coordenate of the middle point
            
            self.prev_cX = self.contourCenterX # save previous contour center X


            if self.getContourCenter(self.MainContour) != 0:
                self.contourCenterX = self.getContourCenter(self.MainContour)[0]
                if abs(self.prev_cX-self.contourCenterX) > 5:
                    self.correctMainContour(self.prev_cX)
            else:
                self.contourCenterX = 0
            
            # self.dir = self.middleX-self.contourCenterX * (areaOfImage/areaOfContour) i don't know why yet 
            self.dir =  int((self.middleX-self.contourCenterX) * self.getContourExtent(self.MainContour))
            print("(",self.middleX, "-", self.contourCenterX,")", "x", self.getContourCenter(self.MainContour))
            print(self.dir)
            
            
            cv2.circle(self.image, (self.contourCenterX, self.middleY), 7, (255,255,255), -1) #cirlces that moves with the contours of line
            cv2.circle(self.image, (self.contourCenterX, self.middleY), 9, (0,0,0), 3) 

            cv2.circle(self.image, (self.middleX, self.middleY), 7, (197,91,111), -1) #circles at the middle point
            cv2.circle(self.image, (self.middleX, self.middleY), 9, (0,0,0), 3) 
            cv2.circle(self.image, (self.middleX, self.middleY), 8, (255,255,255), 1) 
            
            cv2.putText(self.image,str(self.contourCenterX-self.middleX),(self.contourCenterX+20, self.middleY), cv2.FONT_HERSHEY_SIMPLEX, 1,(200,0,200),2)
            cv2.putText(self.image,"Weight:%.3f"%self.getContourExtent(self.MainContour),(self.contourCenterX+20, self.middleY+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200,0,200),1)
        
    def getContourCenter(self, contour):
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return 0
        
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        
        return [x,y]
        
    def getContourExtent(self, contour):
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        rect_area = w*h
        if rect_area > 0:
            return (float(area)/rect_area)
            
    def Aprox(self, a, b, error):
        if abs(a - b) < error:
            return True
        else:
            return False
            
    def correctMainContour(self, prev_cx):
        if abs(prev_cx-self.contourCenterX) > 5:
            for i in range(len(self.contours)):
                if self.getContourCenter(self.contours[i]) != 0:
                    tmp_cx = self.getContourCenter(self.contours[i])[0]
                    if self.Aprox(tmp_cx, prev_cx, 5) == True:
                        self.MainContour = self.contours[i]
                        if self.getContourCenter(self.MainContour) != 0:
                            self.contourCenterX = self.getContourCenter(self.MainContour)[0]
                            
