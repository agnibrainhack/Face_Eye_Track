import cv2
import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
import argparse
import imutils
import dlib
import copy

from utils import *

# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
ds_factor=0.6



def detect_eyes(img, points):

    left_eye = img[points[37][1]-30: points[39][1]+30, points[37][0]-30: points[39][0]+20]
    right_eye = img[points[43][1]-30: points[46][1]+30, points[43][0]-20: points[46][0]+30]
    # print(points[37][1], points[39][1], points[37][0], points[39][0])
    left_eye = cv2.resize(left_eye, (600,400))
    right_eye = cv2.resize(right_eye, (600,400))
    return left_eye, right_eye

def eye_roi(roi):

    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 40, 255, 1)

    kernel = np.ones((3, 3), np.uint8) 
    threshold = cv2.dilate(threshold, kernel)  

    # cv2.imshow("test", gray_roi)
    # cv2.imshow("test1", threshold)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    return roi




def nothing(x):
    pass


def main(single=False, file_name='img.jpg'):
    if not single:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('image')

    counter = 0
    while True:

        if not single:
            _, frame = cap.read()
            cv2.imshow('image', frame)
        else:
            frame = cv2.imread(file_name)
        cv2.imshow('image', frame)
        counter+=1
        if counter %2 !=0 :
            continue

        frame = clean_image(frame)
        # face_frame = complete_aligner(frame)
        face_frame, gen_img_points, points, original = generate_points(frame)
        # face_frame = detect_faces(frame, face_cascade)
        
        
        if len(face_frame) !=0:
            face_frame  = face_frame[0]
            cv2.imshow('face_area',face_frame)
            cv2.imshow('face_with_points',gen_img_points)
            # print(points)
            eyes = detect_eyes(original, points)
            

            for i, eye in enumerate(eyes):
                if eye is not None:
                    eye = eye_roi(eye)
                    eye = cv2.resize(eye, (300,100))
                    cv2.imshow('eye'+str(i), eye)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
