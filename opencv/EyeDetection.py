import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mounth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
bacteria = cv2.imread("bacteria.jpg")
bacteria = cv2.cvtColor(bacteria,cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(bacteria,10,255,cv2.THRESH_BINARY)
invert = cv2.bitwise_not(mask)
def nothing(int):
    pass

cv2.namedWindow("Color")

while cam.isOpened():
    _,frame = cam.read()

    sumMounthX = 0
    sumMounthY = 0
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

        below_eye = eye_gray[sumMounthY:,:]

        mouths = mounth_cascade.detectMultiScale(below_eye,1.3,23,minSize=(50,50))

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

