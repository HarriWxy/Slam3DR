import cv2
import time
import requests
import numpy as np
 
video = 'http://192.168.1.111:4747/video'
 
capture = cv2.VideoCapture(video)
 
while True:
    sucess, img = capture.read()
    cv2.imshow("camera",img)
 
    if cv2.waitKey(1) == 27:
        break
 
capture.release()
cv2.destroyAllWindows()