
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


# In[2]:


def known_face_encoding_to_numpy(path):
    # list file image in folder
    image_list_file = []
    for filename in glob.glob(path + "*"):
        image_list_file.append(filename)
    
    # Show name person in image
    image_names = []
#     image_names = [ntpath.split(name)[1].split('.')[0] for name in image_list_file]
#     image_names = 

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
        
    np.save("../data/numpy/known_face_encoding", known_face_encodings)
    np.save("../data/numpy/known_face_names", known_face_names)
    
    return 1


# In[3]:


start = time.time()
known_face_encoding_to_numpy("/home/damvantai/Documents/face_recognition_demo/data/pictures_of_people_i_know/")
end = time.time()
print("Time encoding image align: ", (end-start))


# In[4]:


known_face_encoding = np.load("../data/numpy/known_face_encoding.npy")


# In[5]:


# print(known_face_encoding)
# print(type(known_face_encoding))
# print(type(known_face_encoding[0]))
# print(type(known_face_encoding[0][0]))
len(known_face_encoding)
known_face_names = np.load("../data/numpy/known_face_names.npy")


# In[6]:


print(len(known_face_names))
print(len(known_face_encoding))
print(known_face_names)

