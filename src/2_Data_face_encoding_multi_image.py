
# coding: utf-8

# In[1]:


import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import face_recognition
from PIL import Image
import glob
import re
import ntpath
import pickle
import time
import os


def known_face_encoding_to_numpy(path):
    # list file image in folder
    image_list_file = []
    for filename in glob.glob(path + "*"):
        image_list_file.append(filename)
    
    # Show name person in image
    image_names = []

#     image_names = [ntpath.split(name)[1].split('.')[0] + ntpath.split(name)[1].split('.')[1] for name in image_list_file]
    image_names = [ntpath.split(name)[1].split('.')[0] for name in image_list_file]
    
    known_face_encodings = []
    known_face_names = []
    
    # Create face encoding for ever image
    for i, filename in enumerate(image_list_file):
        # Load image
        image = face_recognition.load_image_file(image_list_file[i]) 
        face_encoding = face_recognition.face_encodings(image)
        if len(face_encoding) != 0:
            known_face_encodings.append(face_encoding)
            known_face_names.append(image_names[i])
        
        else:
            os.remove(filename)
        
        
#         detector = dlib.get_frontal_face_detector()
#         predictor_68_point_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#         fa_image = FaceAligner(predictor_68_point_model, desiredFaceWidth=160)
        
#         image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         detected = detector(image_gray, 1)
        
#         image_align = fa_image.align(image, image_gray, detected[0])
    
#         # Encoding vector embedding for ever person
#         image_align_encoding = face_recognition.face_encodings(image_align)
#         image_align_encoding = np.asarray(image_align_encoding)
#         image_align_encoding = image_align_encoding.reshape(128,)
#         known_face_encodings.append(image_align_encoding)
#         known_face_names.append(image_names[i])
    
#     len(known_face_encodings)
#     len(known_face_names)
#     print(type(known_face_encodings))
    
#     f = open("data.txt", "w")
#     simplejson.dump(known_face_encodings, f)
#     f.close

    np.save("../data/numpy/known_face_encoding", known_face_encodings)
    np.save("../data/numpy/known_face_names", known_face_names)
    
    # Use pickle
#     with open("../data/data_face_encodings_align.txt", "wb") as f:
#         pickle.dump(known_face_encodings, f)
#     f.close
    
#     with open("../data/data_face_name_align.txt", "wb") as f:
#         pickle.dump(known_face_names, f)
#     f.close
    
#     return known_face_encodings, known_face_names
    return 1


start = time.time()
known_face_encoding_to_numpy("../data/pictures_of_people_i_know/")
end = time.time()
print("Time encoding image align: ", (end-start))


known_face_encoding = np.load("../data/numpy/known_face_encoding.npy")


known_face_encoding = np.load("../data/numpy/known_face_encoding.npy")


len(known_face_encoding)
known_face_names = np.load("../data/numpy/known_face_names.npy")


print(len(known_face_names))
print(len(known_face_encoding))
print(known_face_names)

