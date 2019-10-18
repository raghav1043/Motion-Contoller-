#!/usr/bin/env python
# coding: utf-8

# operate wheelchair through facial expressions
# 
# forward :- open mouth <br/>
# stop :- closed eyes <br/>
# left turn :- kiss <br/>
# right turn :- raised eyebrows <br/>

# In[ ]:


from imutils import face_utils
from utils import *
import numpy as np
import imutils
import dlib
import cv2
import time


# In[ ]:


# Thresholds and consecutive frame length for triggering action.
mouth_ar_thresh = 0.24
mouth_ar_consecutive_frames = 6
eye_ar_thresh = 0.19
eye_ar_consecutive_frames = 4
raised_eyebrows_thresh = 210
raised_eyebrows_consecutive_frames = 4
kiss_thresh = 55
kiss_consecutive_frames = 4


# In[ ]:


# Initialize the frame counters for each action as well as 
# booleans used to indicate if action is performed or not
stop_counter = 0
left_counter = 0
right_counter = 0
forward_counter=0


# In[ ]:


white_color = (255, 255, 255)
yellow_color = (0, 255, 255)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
black_color = (0, 0, 0)


# In[ ]:


# Initialize Dlib's face detector and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


# In[ ]:


#For dlib’s 68-point facial landmark detector:

(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"] 


# In[ ]:


#for eye blink detection code

# Returns eye_aspect_ratio given eye landmarks
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear


# In[ ]:


# for open mouth detection code

# Returns mouth_aspect_ratio given eye landmarks
def open_mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets
    # of vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])

    # Compute the euclidean distance between the horizontal
    # mouth landmarks (x, y)-coordinates
    D = np.linalg.norm(mouth[12] - mouth[16])

    # Compute the mouth aspect ratio
    omar = (A + B + C) / (2 * D)

    # Return the mouth aspect ratio
    return omar


# In[ ]:


# for raised eyeborws detection code

# Returns raised_eyebrows_aspect_ratio given eye landmarks
def raised_eyebrows_aspect_ratio(eyebrows,eye):
    # Compute the euclidean distances between the landmarks (x, y)-coordinates
    A = np.linalg.norm(eyebrows[1] - mouth[1])
    B = np.linalg.norm(eyebrows[3] - mouth[2])

    # Compute the mouth aspect ratio
    rear = (A + B)

    # Return the raised eyebrows aspect ratio
    return rear


# In[ ]:


# for kiss detection code

# Returns kiss_mouth_aspect_ratio given eye landmarks
def kiss_mouth_aspect_ratio(mouth):
    # Compute the euclidean distance between the horizontal
    # mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[0] - mouth[6])

    # Compute the mouth aspect ratio
    kmar = A

    # Return the mouth aspect ratio
    return kmar


# In[ ]:


cap = cv2.VideoCapture(0)
while True:
    
    # load the input image and convert it to grayscale
    _, frame = cap.read()
    # detect faces in the grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    if len(rects)>0:
        rect=rects[0]
        
    else:
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF
        continue
    
    # Determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)  
    
    mouth = shape[mStart:mEnd]
    leftEye = shape[leStart:leEnd]
    rightEye = shape[reStart:reEnd]
    leftEyebrow = shape[lebStart:lebEnd]
    rightEyebrow = shape[rebStart : rebEnd]
    
    
    #open mouth aspect ratio
    omar = open_mouth_aspect_ratio(mouth)
    
    # average eye aspect ratio
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    
    # average raised eyebrows aspect ratio
    leftREAR = raised_eyebrows_aspect_ratio(leftEyebrow,leftEye)
    rightREAR = raised_eyebrows_aspect_ratio(rightEyebrow,rightEye)
    rear = (leftREAR + rightREAR) / 2.0
    
    # kiss mouth aspect ratio
    kmar = kiss_mouth_aspect_ratio(mouth)
    

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, yellow_color, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, yellow_color, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, yellow_color, 1)
        
        
        
    for (x, y) in np.concatenate((mouth, leftEye, rightEye, leftEyebrow, rightEyebrow), axis=0):
        cv2.circle(frame, (x, y), 2, green_color, -1)
    
    
    if omar > mouth_ar_thresh:
        forward_counter += 1
        
        if forward_counter > mouth_ar_consecutive_frames:
            cv2.putText(frame, 'move forward', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
            forward_counter = 0
            
    elif ear < eye_ar_thresh:
        stop_counter += 1
        
        if stop_counter > eye_ar_consecutive_frames:
            cv2.putText(frame, 'stop', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
            stop_counter = 0
            
    elif kmar < kiss_thresh:
        left_counter += 1
        
        if left_counter > kiss_consecutive_frames:
            cv2.putText(frame, 'move left ', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
            left_counter = 0        
        
    elif rear > raised_eyebrows_thresh:
        right_counter += 1
        
        if right_counter > raised_eyebrows_consecutive_frames:
            cv2.putText(frame, 'move right', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
            right_counter = 0
                           

#     print("Image Shown Here",frame)
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cv2.destroyAllWindows()
cap.release()





