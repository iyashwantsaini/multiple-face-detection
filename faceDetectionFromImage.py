import cv2
import numpy as np

# importing_classifiers
face_classifier =cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_classifier  =cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# reading_image
image=cv2.imread('data/Trump.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
# detecting_faces
faces=face_classifier.detectMultiScale(gray,1.5,5)
if faces is ():
    print('No faces detected!')

# faces_found
for (x,y,w,h) in faces:
    
    # draw_rectangle_around_face     
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
#     cv2.imshow('faces',image)
#     cv2.waitKey(0)
    
    # cropping_face_only
    roi_color=image[y:y+h,x:x+w]
    roi_gray=gray[y:y+h,x:x+w]
    
    # eyes_detection
    eyes=eye_classifier.detectMultiScale(roi_gray)
    
    # drawing_eyes_one_by_one
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
#         cv2.imshow('face&eyes',image)
#         cv2.waitKey(0)
    cv2.imshow('faces&eyes',image)
        
cv2.waitKey(0)
cv2.destroyAllWindows()