
import streamlit as st
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import face_recognition
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random

if 'a' not in st.session_state:
    print('initialize')
    st.session_state.a= random.random()

@st.cache(allow_output_mutation=True)
def setup(i):
    tags=[]
    X=[]
    for tag in os.listdir('images'):
        tags.append(tag)
        path=os.path.join('images',tag)
        img=io.imread(os.path.join(path,os.listdir(path)[0]))
        
        face = face_recognition.face_encodings(img)[0]
    
        X.append(face)

    X=np.array(X)
    recognizer=KNeighborsClassifier(n_neighbors=1,algorithm='kd_tree')
    recognizer.fit(X,range(len(X)))
    print('Load successfullt!')
    return [tags,X,recognizer]

tags_X_recognizer=setup(st.session_state.a)

def setup2():
    tags=[]
    X=[]
    for tag in os.listdir('images'):
        tags.append(tag)
        path=os.path.join('images',tag)
        img=io.imread(os.path.join(path,os.listdir(path)[0]))
        
        face = face_recognition.face_encodings(img)[0]
    
        X.append(face)

    X=np.array(X)
    recognizer=KNeighborsClassifier(n_neighbors=1,algorithm='kd_tree')
    recognizer.fit(X,range(len(X)))
    print('Load successfullt!')
    return [tags,X,recognizer]

def process_an_image(img):
    print('Processing image...')
    face_locations = face_recognition.face_locations(img,model='hog')
    faces_feature_vector = face_recognition.face_encodings(img,known_face_locations=face_locations)
    faces_feature_vector=np.array(faces_feature_vector)
    tags_preidcted=tags_X_recognizer[2].predict(faces_feature_vector)

    img_pil=Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for i,face in enumerate(face_locations):
        draw.rectangle(((face[3], face[2]), (face[1],face[0])), outline="white", width=10)
        draw.text((face[3]+10, face[0]+10), tags_X_recognizer[0][tags_preidcted[i]], font=ImageFont.truetype("arial",50))

    img=np.array(img_pil)
    return img


navigation=st.sidebar.selectbox('Navigation',['Home','Add more faces'])
if navigation=='Home':
    st.title('Face Recognition Application')
    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
    img=None
    st.write(str(tags_X_recognizer[0]))
    if uploaded_file is not None:
        img=io.imread(uploaded_file)
        img = (resize(img, (1000, (img.shape[1]*1000) // img.shape[0]))*255).astype('uint8')
        col_one, col_two = st.columns(2)
        col_one.header("Input")
        col_one.image(img)
        
        col_two.header("Output")
        #dosth
        img_out=process_an_image(img)
        col_two.image(img_out)
elif navigation=='Add more faces':
    st.title('Face Recognition Application')
    name=st.text_input(label="Name")
    uploaded_file = st.file_uploader("Upload portrait image",type=['png','jpeg','jpg'])

    if st.button('Add'):
        if (uploaded_file is not None) and (name != '') :
            if name not in os.listdir('images'):
                os.makedirs(os.path.join('images',name))
                img=io.imread(uploaded_file)
                io.imsave(os.path.join('images',name,'portrait.png'),img)
                st.session_state['a']=random.random()
                st.success("Add successfully, Please clear the cache and rerun the app again!")

            else:
                st.error("Please select another name")
        else:
            st.error("Blank name or image file, please check again!!")
