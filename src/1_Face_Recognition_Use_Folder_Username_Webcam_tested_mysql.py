
# coding: utf-8

# In[14]:


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
import sqlite3
from imutils.face_utils import rect_to_bb
import imutils
from imutils import face_utils
import os
import shutil
import inception_resnet_v1
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pymysql as MySQLdb
import math
from imutils.video import VideoStream
# In[15]:


# show datetime now
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# In[16]:


# folder unknown_people through day
folder_name = "unknown_people"+date


# In[17]:


# create folder unknown_people in day
os.mkdir("../data/" + folder_name)


# In[18]:


# connect mysql 3 table data: people_known, people_inroom, people_unknown
people = MySQLdb.connect(host='localhost',
                      port=3306,
                      user='root',
                      passwd='hanoi1994',
                      db='people')
c = people.cursor()
c.execute('select * from people_known order by old')
for row in c:
    print(row)
c.execute('delete from people_inroom')
c.execute('select * from people_inroom order by old')
for row in c:
    print(row)
c.execute('delete from people_unknown')
c.execute('select * from people_unknown order by old')
for row in c:
    print(row)

c.execute('select * from history_come')
for row in c:
    print(row)
c.execute('delete from history_come')


# In[19]:


# recognition face center
def face_center(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # detect rectangles from image faces
    rects = detector(image_gray, 1)
    for (i, rect) in enumerate(rects):

        points = predictor(image_gray, rect)
        points = face_utils.shape_to_np(points)
        center_face = math.fabs(points[36][1] - points[45][1])
        print("center face: ", center_face)

        left = points[36][0] - points[33][0]
        right = points[33][0] - points[45][0]
        bias_horizonal = math.fabs(left - right)
        print("bias_horizonal: ", bias_horizonal)

        # top = points[19][1] - points[33][1]
        # bottom = points[33][1] - points[8][1]
        # bias_vertical = math.fabs(top - bottom)
        # print("bias_vertical: ", bias_vertical)

        # if center_face < 13 and center_face > -13 and bias_horizonal < 13 and bias_vertical < 35:
        if center_face < 10 and bias_horizonal < 13:
        	# x, y = points[33]
         #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
         #    x, y = points[36]
         #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
         #    x, y = points[45]
         #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
         #    x, y = points[19]
         #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
         #    x, y = points[31]
         #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
         #    x, y = points[8]
         #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            return True
        else:
            return False


# In[20]:


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


# In[21]:


# Load model and run graph inception resnet v1 from models and file resnetv1_inception.py
tf.reset_default_graph() 
# sess = tf.InteractiveSession()
sess = tf.Session()
images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
train_mode = tf.placeholder(tf.bool)
age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                             phase_train=train_mode,
                                                             weight_decay=1e-5)

gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state("../gender_age_tf/models/")
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("restore model!")
else:
    pass


# In[22]:


face_locations = [] # locations of all faces in image
face_encodings = [] # vecto encodings of all faces in image

name = ""

total_in_room = 0
total_known_in_room = 0
index = 0
faces = np.empty((1,160, 160, 3))

# Load known_face_encoding_array and known_face name from npy
known_face_encodings_array = np.load("../data/numpy/known_face_encoding.npy")
known_face_names = np.load("../data/numpy/known_face_names.npy")

# Convert nparray -> list 
number_person = len(known_face_encodings_array)
known_face_encodings_array = known_face_encodings_array.reshape(number_person, 128)
known_face_encodings = []
for i in range(len(known_face_encodings_array)):
    known_face_encodings.append(known_face_encodings_array[i])

# face encodings of people in room
face_encodings_in_room = []
face_names_in_room = []

# face encodings of people unknown in room
face_encodings_unknown_in_room = []
face_names_unknown_in_room = []

# lib detect face in image
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# lib aligner face in image 
fa = FaceAligner(predictor, desiredFaceWidth=160)


# In[ ]:
video_capture = VideoStream(src=0).start()
video_capture_extern = VideoStream(src=1).start()
# Capture from camera of own computer
# video_capture = cv2.VideoCapture(0)
# video_capture_extern = cv2.VideoCapture(1)

while True:
    # COME IN
    # Grab a single frame of video
    # ret, frame = video_capture.read()

    now = datetime.datetime.now()
    now = str(now).split(' ')[1]

    frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Use loop for ever face in image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_unknown = frame[top:bottom, left:right]
        frame_gray = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2GRAY)
        
        # test face center to process
        if face_center(face_unknown) == True:
            # compute distance between face come in with all face known
            distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            if (len(distance) > 0):
	            point = np.min(distance)
	            print("point known face encoding in face center ", point)
	            index_point_min = np.argmin(distance)
	            
	            if point > 0.4:
	                if os.path.isfile("../data/" + folder_name + "/unknown_0.jpg"):
	                    distance_unknown = face_recognition.face_distance(face_encodings_unknown_in_room, face_encoding)
	                    min_distance_unknown = np.min(distance_unknown)
	                    index_min_distance_unknown = np.argmin(distance_unknown)
	                    
	                    # min_distance > 0.45 unknown people new
	                    if min_distance_unknown > 0.4:
	                        a = os.listdir("../data/" + folder_name + "/")
	                        a.sort()
	                        index = int(a[-1].split('.')[0].split('_')[1])
	                        index += 1
	                        path = "../data/" + folder_name + "/unknown_" + str(index) + ".jpg"
	                        name = "unknown_" + str(index)
	                        cv2.imwrite(path, face_unknown)
	                        face_encodings_unknown_in_room.append(face_encoding)
	                        face_encodings_in_room.append(face_encoding)
	                        face_names_in_room.append(name)
	                        face_names_unknown_in_room.append(name)
	                        print(name)

	                        # Draw a rectangle with detect the face
	                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	                        # Draw a label with a name below the face
	                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	                        font = cv2.FONT_HERSHEY_DUPLEX
	                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	                        # Estimate gender and age
	                        # Convert to rect
	                        face_detected = _css_to_rect((top, right, bottom, left))
	                        print(face_detected)
	                        faces[0, :, :, :] = fa.align(rgb_small_frame, frame_gray, face_detected)
	                        age_predict, gender_predict = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
	                        
	                        label = "{}, {}".format(int(age_predict), "F" if gender_predict == 0 else "M")
	                        cv2.putText(frame, label, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

	                        c.execute("insert into people_inroom values (%s, %s, %s)", (name, int(gender_predict), int(age_predict)))
	                        people.commit()
	                        c.execute("insert into people_unknown values (%s, %s, %s)", (name, int(gender_predict), int(age_predict)))
	                        people.commit()

	                    else:                      
	                        name = "unknown_" + str(index_min_distance_unknown)
	                        # Draw a rectangle with detect the face
	                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	                        # Draw a label with a name below the face
	                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	                        font = cv2.FONT_HERSHEY_DUPLEX
	                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	                        c.execute("select * from people_unknown where name = '%s'" % name)
	#                         print("show c.fetchone()", c.fetchone())
	                        name, gender_predict, age_predict = c.fetchone()
	                        label = "{}, {}".format(int(age_predict), "F" if gender_predict == 0 else "M")
	                        cv2.putText(frame, label, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

	                else:
	                    name = "unknown_0"
	                    path = "../data/" + folder_name + "/unknown_0" + ".jpg"
	                    cv2.imwrite(path, face_unknown)
	                    face_encodings_unknown_in_room.append(face_encoding)
	                    face_encodings_in_room.append(face_encoding)
	                    face_names_in_room.append(name)
	                    face_names_unknown_in_room.append(name)
	                    print(name)

	                    # Draw a rectangle with detect the face
	                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	                    # Draw a label with a name below the face
	                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	                    font = cv2.FONT_HERSHEY_DUPLEX
	                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	                    # estimate gender and age
	                    face_detected = _css_to_rect((top, right, bottom, left))
	#                     face_detected = detector(face_unknown, 1)
	                    
	                    faces[0, :, :, :] = fa.align(rgb_small_frame, frame_gray, face_detected)
	                    age_predict, gender_predict = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
	                    label = "{}, {}".format(int(age_predict), "F" if gender_predict == 0 else "M")
	                    cv2.putText(frame, label, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

	                    c.execute("insert into people_inroom values (%s, %s, %s)", (name, int(gender_predict), int(age_predict)))
	                    people.commit()
	                    c.execute("insert into people_unknown values (%s, %s, %s)", (name, int(gender_predict), int(age_predict)))
	                    people.commit()

	            else:

	                face_encodings_in_room.append(face_encoding)
	                name = known_face_names[index_point_min]
	                face_names_in_room.append(name)
	                # Draw a rectangle with detect the face
	                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	                # Draw a label with a name below the face
	                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	                font = cv2.FONT_HERSHEY_DUPLEX
	                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	                print(name)
	                c.execute("select * from people_known where name= '%s'" % name)
	                data_temp = c.fetchone()  # name in people known
	#                 data_temp[1] = ord(data_temp[1])
	                c.execute("select * from people_inroom where name='%s'" % name)
	                data_name_inroom = c.fetchone() # name in room
	                if data_name_inroom == None:
	                    c.execute("insert into people_inroom values (%s, %s, %s)", (data_temp[0], ord(data_temp[1]), data_temp[2]))
	                    people.commit()
                    
    
    total_in_room = c.execute("SELECT name FROM people_inroom")
    total_known_in_room = total_in_room - c.execute("SELECT name FROM people_unknown")
    print("total in people in room ", total_in_room)
    print("total in people known name in room ", total_known_in_room)
        
    # # Display the resulting image
    # ret_extern, frame_extern = video_capture_extern.read()
    
    # # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_frame_extern = frame_extern[:, :, ::-1]
    

    # # Find all the faces and face encodings in the current frame of video
    # face_locations = face_recognition.face_locations(rgb_frame_extern)
    # face_encodings = face_recognition.face_encodings(rgb_frame_extern, face_locations)

    # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    #     face_unknown = frame_extern[top:bottom, left:right]
    #     frame_gray = cv2.cvtColor(rgb_frame_extern, cv2.COLOR_BGR2GRAY)
        
    #     # test face center to process
    #     if face_center(face_unknown) == True:

    #         distance = face_recognition.face_distance(face_encodings_in_room, face_encoding)
    #         if len(distance > 0):
    #             point = np.min(distance)
    #             index_point_min = np.argmin(distance)
    #             name = face_names_in_room[index_point_min]
                
    #             # Draw a rectangle with detect the face
    #             cv2.rectangle(frame_extern, (left, top), (right, bottom), (0, 0, 255), 2)

    #             # Draw a label with a name below the face
    #             cv2.rectangle(frame_extern, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #             font = cv2.FONT_HERSHEY_DUPLEX
    #             cv2.putText(frame_extern, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                
    #             c.execute("delete from people_inroom where name='%s'" % name)
    #             people.commit()
    
    cv2.imshow('Video', frame)
    # cv2.imshow('Video1', frame_extern)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
people.close()

