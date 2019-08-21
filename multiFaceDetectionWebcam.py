import cv2
import numpy as np

# importing_haarcascade_classifiers
face_classifier =cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_classifier  =cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# face_detection_function
def face_detection(image):
    
    # grascaling_image_passed
    if ret is True:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        # detecting_faces    
        faces=face_classifier.detectMultiScale(gray,1.6,7)

        # drawing_face_rectangle
        for (x,y,w,h) in faces:
            
            # draw_rectangle_around_face     
            cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
            
            # cropping_face_only
            roi_color=image[y:y+h,x:x+w]
            roi_gray=gray[y:y+h,x:x+w]
            
            # eyes_detection
            eyes=eye_classifier.detectMultiScale(roi_gray)
            
            # drawing_eyes_rectangles
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
    
    # returning_image_with_rectangles
    return image
    
# capturing_video_from_webcam
cap=cv2.VideoCapture(0)

while True:

    # reading_from_camera
    ret,frame=cap.read()
    cv2.imshow('face_detection',face_detection(frame))

    # if_enter_pressed_then_exit
    if cv2.waitKey(1)==13:
        break
        
# releasing_camera
cap.release()
# destroying_window
cv2.destroyAllWindows()
