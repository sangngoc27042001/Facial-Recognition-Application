import face_recognition
import matplotlib.pyplot as plt
from skimage import io
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import argparse

tags=[]
X=[]
recognizer=None
def setup():
    global X
    global recognizer
# 1. Loading all the files containing the potrait images of each person.
    for tag in os.listdir('images'):
        tags.append(tag)
        path=os.path.join('images',tag)
        img=io.imread(os.path.join(path,os.listdir(path)[0]))
        
# 2. Extract feature from the potrait images
        face = face_recognition.face_encodings(img)[0]
    
# 3. Gather all feature vectors to a matrix
        X.append(face)
# 4. Convert to numpy array and fit the KNN model
    X=np.array(X)
    recognizer=KNeighborsClassifier(n_neighbors=1,algorithm='kd_tree')
    recognizer.fit(X,range(len(X)))
    
# process an new input image: 
# img_file_in: file path of the input image
# img_file_out: file path of the output image
def process_an_image(img_file_in,img_file_out):
# 1. Read the image in numpy array form
    img=io.imread(img_file_in)

# 2. Face Detection
    face_locations = face_recognition.face_locations(img,model='hog')

# 3. Feature Extraction
    faces_feature_vector = face_recognition.face_encodings(img,known_face_locations=face_locations)
    faces_feature_vector=np.array(faces_feature_vector)

# 4. Face Recognition using KNN model
    tags_preidcted=recognizer.predict(faces_feature_vector)

# 5. Draw bounding boxs, tags for image.
    for i,face in enumerate(face_locations):
        cv2.rectangle(img, (face[3], face[2]), (face[1],face[0]), (255, 255, 255), 5)
        cv2.putText(img, tags[tags_preidcted[i]], (face[3], face[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
    io.imsave(img_file_out,img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    setup()
    process_an_image(args.input,args.output)