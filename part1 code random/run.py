import os
import cv2
import numpy as np
from numpy import asarray
from scipy.spatial.distance import cosine
from sklearn import preprocessing
import tensorflow as tf
from tkinter import *
import pandas as pd
import time

df = pd.read_excel('sample medicine drug.xlsx')
resnet_model = tf.lite.Interpreter('faceRecgnitionresnet.tflite')
resnet_model.allocate_tensors()
input_details = resnet_model.get_input_details()
output_details = resnet_model.get_output_details()

print("\n------------------------------- Model Loaded -------------------------------")

dataset_names = os.listdir('dataset')
le = preprocessing.LabelEncoder()
labelencoding= le.fit_transform(dataset_names)

    
CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def screen(x):
    text="Welcome " + x + '\n' + "Your Medicine is: "+ med_name
    print(text)
    # Wait for 5 seconds
    time.sleep(5)
    


def detect_faces(image):
    """
    It takes an image as input, converts it to grayscale, and then detects faces in the image. 
    
    The function returns the face image and the coordinates of the face in the original image
    
    :param image: The image to detect faces in
    :return: The cropped image and the coordinates of the bounding box.
    """
   
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
    for x,y,w,h in faces:
        sub_img=image[y-10:y+h+10,x-10:x+w+10]
        sub_img=cv2.resize(sub_img,(224,224))
    return sub_img, (x-10,y-10, x+w+10, y+h+10) 

 
def get_embeddings(faces):
    """
    It takes a list of faces and returns a list of embeddings
    
    :param faces: a list of faces
    :return: The embeddings of the face.
    """
    samples = asarray(faces, 'float32')
    samples = np.array(samples, ndmin=4)
    
    resnet_model.set_tensor(input_details[0]['index'], samples)
    resnet_model.invoke()
    pred = resnet_model.get_tensor(output_details[0]['index'])

    return pred[0][0][0]


img_class_list=[]
train_embedding = []
# Iterating through the list of names in the dataset folder.
for i in dataset_names:
    if i == '.DS_Store':
      pass
    else:
      player_image = os.listdir(f'dataset/{i}')
      for j in player_image:
        image_path = f'dataset/{i}/{j}'
        print(image_path)
        img_class=image_path.split('/')[-2]
        img_class_list.append(img_class)
        image = cv2.imread(image_path)
        train_embedding_=get_embeddings(detect_faces(image)[0])
        train_embedding.append(train_embedding_)


y_train_embedding= le.fit_transform(img_class_list) 
from sklearn.linear_model import LogisticRegression as LR

model = LR(multi_class='multinomial', solver='lbfgs')
model.fit(train_embedding, y_train_embedding)

print("\n------------------------------- Model Training Done -------------------------------")

# org
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
fps = cap.get(cv2.CAP_PROP_FPS)
print("\n------------------------------- Streaming Now -------------------------------")


# The next code is detecting the face and then comparing it with the faces in the database. If the
# face is in the database, it will display the name of the person. If the face is not in the database,
# it will display unknown.
person_get = True
while True:
      ret_val, frame = cap.read()
      if ret_val:
        if person_get == True:
          try:
            test_img_face = detect_faces(frame)[0]
            test_img_face= np.expand_dims(test_img_face,axis=0)
            X_test_embedding = get_embeddings(test_img_face)
            X_test_embedding= np.expand_dims(X_test_embedding,axis=0)
            y = model.predict(X_test_embedding)
            z = detect_faces(frame)[1]
            frame = cv2.rectangle(frame, pt1=tuple(z[0:2]), pt2=tuple(z[2:]), color=(255,0,0), thickness=10)
            org = (z[0],z[1]-15)
            r_list = []
            for i in range (len(train_embedding)):
              r = cosine(train_embedding[i] ,X_test_embedding[0])
              print(r)
              r_list.append(r)
            
            r=min(r_list)
            if r<0.8:
                person = le.inverse_transform(y)
                #createGUI(person)
               
                med_name = df['English name'].sample().iloc[0]
                screen(person[0])
                cap.release()
                cv2.destroyAllWindows()
                person_get = False
#                 cap = cv2.VideoCapture(0)

            else:
                person = 'unknown'
            print(person)
            frame = cv2.putText(frame, person, org, font,fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow('frame',frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
          except:
            person = 'unknown'
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
          cv2.imshow('frame',frame)
          person_get = True
      else:
        break
cap.release()
cv2.destroyAllWindows()