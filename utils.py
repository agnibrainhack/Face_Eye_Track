import cv2
import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
import argparse
import imutils
import dlib
import copy

def clean_image(im):
    return im

def complete_aligner(image):
    args = {}
    args["shape_predictor"] = 'shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    fa = FaceAligner(predictor, desiredFaceWidth=480)
    image = imutils.resize(image, width=480)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

        # cv2.imshow("Original", faceOrig)
        # cv2.imshow("Aligned", faceAligned)

    return faceAligned

def generate_points(image):

    args = {}
    detector = dlib.get_frontal_face_detector()
    args["shape_predictor"] = 'shape_predictor_68_face_landmarks.dat'
    # args['image'] = 'image/img.jpg'
    predictor = dlib.shape_predictor(args["shape_predictor"])
    # image = cv2.imread(args["image"])
    image = imutils.resize(image, width=500)
    
    original = copy.copy(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    
    face_area = [] #@
    point = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_area.append(original[y-20:y+h+40,x-20:x+w+20])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            point.append((int(x), int(y)))

    return face_area, image, point, original
        # nparray = np.asarray(point)
        # np.savetxt("points.txt", nparray, fmt='%d')
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)




def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame
