import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mounth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
eyeCount = 0
cv2.namedWindow("Eyes", cv2.WINDOW_AUTOSIZE)


def nothing(int):
    pass

while cam.isOpened():
    _,frame = cam.read()

    sumMounthX = 0
    sumMounthY = 0
    currentEye = []
    mask_wearing = False

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    img = frame.copy()

    faces = face_cascade.detectMultiScale(gray_frame,1.25,10)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
        eye_gray = gray_frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(eye_gray,1.25,10)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(x+ex,y+ey), (x+ew+ex,y+eh+ey),(0,0,0),2)
            sumMounthX = ex+ew
            sumMounthY = eh+ey
            if(eyeCount == 0):
                #ALL
                currentEye = frame[y+ey:y+sumMounthY, x+ex:x+sumMounthX]
                cv2.imshow("Eyes", currentEye)
                # Get the eye Color
                eyeImg = currentEye.copy()
                currentEye = cv2.GaussianBlur(eyeImg, (5, 5), 1)
                eyeGray = cv2.cvtColor(currentEye, cv2.COLOR_BGR2GRAY)

                minDist = currentEye.shape[0] / 8
                param1 = 60
                param2 = 40

                circles = cv2.HoughCircles(eyeGray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                           minRadius=0, maxRadius=60)

                circles = np.uint16(np.around(circles))

                if circles is not None:
                    for x, y, r in circles[0, :]:
                        # cv2.circle(img, (x, y), r, (0, 255, 0), 3)
                        cv2.circle(eyeImg, (x, y), 2, (0, 0, 255), 1)
                cv2.imshow("eye", eyeImg)

        eyeCount = 1

        below_eye = eye_gray[sumMounthY:,:]

        mouths = mounth_cascade.detectMultiScale(below_eye,1.05,4)

        for (mx,my,mw,mh) in mouths:
            cv2.rectangle(frame,(x+mx,my+sumMounthY+y), (x+mx+mw,my+sumMounthY+y+mh),(0,0,0),2)
            mask_wearing = True

    if(mask_wearing):
        print("maske takılıyor")
    else:
        print("maske takılmıyor")



    cv2.imshow("img",frame)

    key = cv2.waitKey(5)
    if key == ord("q"):
        break

cv2.destroyAllWindows()

