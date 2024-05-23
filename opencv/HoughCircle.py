import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd

image = cv2.imread("goku.jpeg")
img = image.copy()
image = cv2.GaussianBlur(img,(5,5),1)
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

minDist = image.shape[0] / 8
param1 = 60
param2= 40

circles = cv2.HoughCircles(imageGray,cv2.HOUGH_GRADIENT,1,minDist,param1=param1,param2=param2,
                           minRadius=0, maxRadius=60)
circles = np.uint16(np.around(circles))
if circles is not None:
    for x,y,r in circles[0,:]:
        cv2.circle(img,(x,y),r,(0,255,0),3)
        cv2.circle(img,(x,y),2,(0,0,255),4)

cv2.imshow("deat", img)
cv2.waitKey(0)
cv2.destroyAllWindows()