# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:15:32 2020

@author: Faraz
"""


import cv2
import time


cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

while(video.isOpened()):
    ret,frame= video.read()
#print(ret)
#print(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=20,minSize=(30,30))
    #how much image size to reduce, how many neigh each rectangle should have, what frame of image should be
    
    for(x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #save detected face on device add system time to save multiple images
        timestr = time.strftime("%Y%m%d-%H%M%S")
        #print (timestr)
        sub_face=frame[y:y+h,x:x+w]
        file_name="face"+timestr+".jpg"
        cv2.imwrite(file_name,sub_face)
   
    cv2.imshow("video",frame)   

    key=cv2.waitKey(1)
    if key==ord("q"):
        break

video.release()
cv2.destroyAllWindows()