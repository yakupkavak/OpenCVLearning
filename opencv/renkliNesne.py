import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd

cam = cv2.VideoCapture(0)

def nothing(int):
    pass

cv2.namedWindow("Color")
cv2.createTrackbar("H1","Color",0,359,nothing)
cv2.createTrackbar("H2","Color",0,359,nothing)
cv2.createTrackbar("S1","Color",0,255,nothing)
cv2.createTrackbar("S2","Color",0,255,nothing)
cv2.createTrackbar("V1","Color",0,255,nothing)
cv2.createTrackbar("V2","Color",0,255,nothing)


kernel = np.ones((5,5),np.uint8)
font = cv2.FONT_ITALIC
while cam.isOpened():

    _,frame = cam.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    img = frame.copy()

    h1 = int(cv2.getTrackbarPos("H1","Color")/2)
    h2 = int(cv2.getTrackbarPos("H2","Color")/2)
    s1 = cv2.getTrackbarPos("S1","Color")
    s2 = cv2.getTrackbarPos("S2","Color")
    v1 = cv2.getTrackbarPos("V1","Color")
    v2 = cv2.getTrackbarPos("V2","Color")

    lower = np.array([h1,s1,v1])
    upper = np.array([h2,s2,v2])
    mask = cv2.inRange(hsv,lower,upper)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel=kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel=kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 50000 or area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        print(x,y,w,h)
        color = (rnd(0,  256), rnd(0, 256), rnd(0, 256))


        try:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img,ellipse,color,-1)

        except cv2.error as er:
            print(er)


        #cv2.drawContours(img, contours, i, color, -1,cv2.LINE_8, hierarchy , 0)
        text = str((w,h))
        cv2.putText(img,text,(x,y),font,1,color,2)



    res = cv2.bitwise_and(frame,frame,mask=mask)

    #cv2.imshow("mask",mask)
    cv2.imshow("img",img)











    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()

