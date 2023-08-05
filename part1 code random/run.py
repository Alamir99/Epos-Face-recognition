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
from sklearn.linear_model import LogisticRegression as LR
import random

from verification import Verify

import pandas as pd

def percise_medicine():
    # Define a dictionary mapping diseases to medicines
    disease_to_medicine = {
        'episodic palpitations': 'amiodarone'
    }
    check_history = {'amiodarone': ['x', 'y', 'z']}
    check_symptoms = {'amiodarone': ['q', 'w', 'e']}
    
    # Read the DataFrame from CSV
    df = pd.read_csv('database.csv')
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the disease and recommended medicine for the current row
        disease = row['chronic_disease']
        recommended_medicine = row['recommended_medicine']
        history = row['history']
        symptoms = row['symptoms']
        
        # Check if the recommended medicine exists in the check_history dictionary
        if recommended_medicine in check_history:
            # Get the history values for the recommended medicine
            history_values = check_history[recommended_medicine]
            # Check if any history value is not present in the 'history' column
            if history in history_values:
                print(f"Conflict detected for disease: {disease}. Medicine: {recommended_medicine}. History: {row['history']}")
        
        # Check if the recommended medicine exists in the check_symptoms dictionary
        if recommended_medicine in check_symptoms:
            # Get the symptom values for the recommended medicine
            symptom_values = check_symptoms[recommended_medicine]
            
            # Check if any symptom value is not present in the 'symptoms' column
            if symptoms in symptom_values:
                print(f"Conflict detected for disease: {disease}. Medicine: {recommended_medicine}. Symptoms: {row['symptoms']}")
        
        # Update the 'recommended_medicine' column if it was originally 'None'
        if recommended_medicine == 'None':
            df.at[index, 'recommended_medicine'] = disease_to_medicine.get(disease, 'None')
    
    # Save the updated DataFrame to CSV
    df.to_csv('database.csv', index=False)


class FaceRecognizer:
    def __init__(self):
        self.verify = Verify()
        self.qdf = pd.read_csv('database.csv')
        self.resnet_model = tf.lite.Interpreter('faceRecgnitionresnet.tflite')
        self.resnet_model.allocate_tensors()
        self.input_details = self.resnet_model.get_input_details()
        self.output_details = self.resnet_model.get_output_details()
        self.dataset_names = os.listdir('dataset')
        self.le = preprocessing.LabelEncoder()
        self.labelencoding = self.le.fit_transform(self.dataset_names)
        self.CASCADE = "Face_cascade.xml"
        self.FACE_CASCADE = cv2.CascadeClassifier(self.CASCADE)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 0, 0)
        self.thickness = 2
        self.cap = cv2.VideoCapture(0) 
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.person_get = True
        self.train_embedding = []
        self.model = None
    
    def screen(self, x, med_name,email):
        random_code = random.randint(1, 100)
        sender_email = "aliaxistower@hotmail.com"
        sender_password = "Dejavu12345"
        recipient_email = email 
        subject = "Verification Email"
        body = f"Hello {x},\n\n"
        body += "This is a verification message to confirm your account. We appreciate your interest in our services.\n"
        body += f"Please take a moment to review the recommended medicine based on your medical history {med_name}.\n"
        body += "Your health is important to us, and we strive to provide you with the best possible care.\n"
        body += f"Please share the following code with the cashier for verification: {random_code}. If you have any questions or need further assistance, please don't hesitate to reach out to our support team.\n\n"
        body += "Thank you for choosing our service.\n\n"
        body += "Best regards,\n"
        body += "Your Healthcare Provider"
        self.verify.send_verification_email(sender_email, sender_password, recipient_email, subject, body)
        # Verification process
        while True:
            user_code = input("Please enter the verification code sent to your email: ")
            if int(user_code) == random_code:
                print("Verification successful. Please wait for 5 seconds.")
                text = "Welcome " + x + '\n' + "Your Medicine is: " + med_name
                print(text)
                break
            else:
                print("Verification code is incorrect. Please try again.")

    def detect_faces(self, image):
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0)
        for x, y, w, h in faces:
            sub_img = image[y - 10:y + h + 10, x - 10:x + w + 10]
            sub_img = cv2.resize(sub_img, (224, 224))
        return sub_img, (x - 10, y - 10, x + w + 10, y + h + 10)

    def get_embeddings(self, faces):
        samples = asarray(faces, 'float32')
        samples = np.array(samples, ndmin=4)

        self.resnet_model.set_tensor(self.input_details[0]['index'], samples)
        self.resnet_model.invoke()
        pred = self.resnet_model.get_tensor(self.output_details[0]['index'])

        return pred[0][0][0]

    def train_model(self):
        img_class_list = []
        for i in self.dataset_names:
            if i == '.DS_Store':
                pass
            else:
                player_image = os.listdir(f'dataset/{i}')
                for j in player_image:
                    if j.endswith(('.png', '.jpg')):
                        image_path = f'dataset/{i}/{j}'
                        print(image_path)
                        img_class = image_path.split('/')[-2]
                        img_class_list.append(img_class)
                        image = cv2.imread(image_path)
                        try:
                            train_embedding_ = self.get_embeddings(self.detect_faces(image)[0])
                            self.train_embedding.append(train_embedding_)
                        except:
                            pass


        y_train_embedding = self.le.fit_transform(img_class_list)
        self.model = LR(multi_class='multinomial', solver='lbfgs')
        self.model.fit(self.train_embedding, y_train_embedding)

        print("\n------------------------------- Model Training Done -------------------------------")

    def run(self):
        while True:
            ret_val, frame = self.cap.read()
            if ret_val:
                if self.person_get:
                    try:
                        test_img_face = self.detect_faces(frame)[0]
                        test_img_face = np.expand_dims(test_img_face, axis=0)
                        X_test_embedding = self.get_embeddings(test_img_face)
                        X_test_embedding = np.expand_dims(X_test_embedding, axis=0)
                        y = self.model.predict(X_test_embedding)
                        z = self.detect_faces(frame)[1]
                        frame = cv2.rectangle(frame, pt1=tuple(z[0:2]), pt2=tuple(z[2:]), color=(255, 0, 0), thickness=10)
                        org = (z[0], z[1] - 15)
                        r_list = []
                        for i in range(len(self.train_embedding)):
                            r = cosine(self.train_embedding[i], X_test_embedding[0])
                            # print(r)
                            r_list.append(r)

                        r = min(r_list)
                        if r < 0.5:
                            person = self.le.inverse_transform(y)
                            filtered_df = self.qdf[self.qdf['name'] == person[0]]['recommended_medicine']
                            email_df = self.qdf[self.qdf['name'] == person[0]]['email']
                            if not filtered_df.empty:
                                med_name = filtered_df.iloc[0]
                                email = email_df.iloc[0]
                            else:
                                med_name = 'unknown'
                                email = 'unknown'
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return person[0], med_name,email
                            
                            self.person_get = False
                        else:
                            person = 'unknown'
                        # print(person)
                        frame = cv2.putText(frame, person, org, self.font, self.fontScale, self.color, self.thickness,
                                            cv2.LINE_AA)

                        cv2.imshow('frame', frame)
                        if cv2.waitKey(2) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        #print(f"Error: {str(e)}")
                        person = 'unknown'
                        cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    cv2.imshow('frame', frame)
                    self.person_get = True
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return person[0], med_name,email

if __name__ == "__main__":
    percise_medicine()
    recognizer = FaceRecognizer()
    recognizer.train_model()
    person, med_name ,email = recognizer.run()
    recognizer.screen(person, med_name,email)



