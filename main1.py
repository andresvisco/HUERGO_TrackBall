import cv2
import imutils
from collections import deque
import time
import numpy as np
import json


#video = cv2.VideoCapture('videoplayback_Trim.mp4')
video=cv2.VideoCapture(0)


with open('data.json') as f:
    data=json.load(f)


greenLower = (data["data"]["low"]["h"],data["data"]["low"]["s"],data["data"]["low"]["v"])
greenUpper = (data["data"]["high"]["h"],data["data"]["high"]["s"],data["data"]["high"]["v"])
pts=deque(maxlen=128)
time.sleep(2.0)


while True:
    ret, frame = video.read()
    if ret == True:
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None
        h=450
        w=600
        image=np.zeros((h,w,3), np.uint8)
        if len(cnts)>0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
            if radius>10:
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 1)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                #cv2.ellipse(frame,(int(x), int(y)),(180,90),0,0,420,(0, 255, 255),1)

        
        pts.appendleft(center)
        linea = None
    
        color=[255,255,255]
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            thickness=int(np.sqrt(64/float(i+1))*2.0)
    #        image[pts[i]]=color
            cv2.line(frame, pts[i-1],pts[i],(0,0,255), thickness)
        frame=cv2.flip(frame, 1)

        cv2.imshow('frame', frame)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
video.destroyAllWindows()