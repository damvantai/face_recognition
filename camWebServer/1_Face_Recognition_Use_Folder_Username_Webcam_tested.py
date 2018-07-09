
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import face_recognition
from PIL import Image
import glob
import re
import ntpath
import os
import dlib
import time
from imutils.face_utils import FaceAligner


# In[2]:

def facename_recognition():
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    name = ""
    frame_number = 0

    # Load known_face_encoding_array and known_face name from npy
    known_face_encodings_array = np.load("../data/numpy/known_face_encoding.npy")
    known_face_names = np.load("../data/numpy/known_face_names.npy")

    # Convert nparray -> list 
    number_person = len(known_face_encodings_array)
    known_face_encodings_array = known_face_encodings_array.reshape(number_person, 128)
    known_face_encodings = []
    for i in range(len(known_face_encodings_array)):
        known_face_encodings.append(known_face_encodings_array[i])

        
    # Capture from camera of own computer
    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        # Frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame_number += 1
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = frame[:, :, ::-1]
            
        # Use computer configuration low
    #     if (frame_number % 10 == 0):
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
    #         print(face_locations)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

    #             print(face_recognition.face_distance(known_face_encodings, face_encoding))
                print(matches)
                print(np.min(face_recognition.face_distance(known_face_encodings, face_encoding)))
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
    #                     print(name)
                face_names.append(name)
                print(name)
    #     process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #         top *= 4
    #         right *= 4
    #         bottom *= 4
    #         left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

