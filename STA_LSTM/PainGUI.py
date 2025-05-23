
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
#from transform_2d import est_similarity_trans, similarity_trans
from transform3d_scal import est_similarity_trans, similarity_trans
from keypoints_extract_3d import keypoints_extract
from cv2_msg_display import msg_display

## library for inference

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score
from data import data_preprocess, data_trans
from modelbase import STA_LSTM as Net
from sklearn.metrics import confusion_matrix

LSTM_IN_DIM =956
SEQUENCE_LENGTH = 30  # 2
IN_DIM = LSTM_IN_DIM* SEQUENCE_LENGTH  # always use 60 / base number to get the final datapoints, e.g. 60 / base_2 = 30

LSTM_HIDDEN_DIM = 64  # 32 #64
OUT_DIM = 3
text = "No Pain"

net = torch.load('C:\\projects\\STALSTM\\models\\manual_label_all_e60.pth')

net.eval()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def open_dialog_box():
    text = "No Pain"
    i = 0  # count of feature vector
    keypoints = np.zeros(shape=(478, 3))
    keypoints_frame = np.zeros(shape=(1, 28680))  # create a zeros numpy array to store keypoints corrdinates for 30 frames 468x2x30 = 28080
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # refine landmarks using attention
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    # continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ##############################################
                        ## save landmarks (lm.x, lm.y) into an array
                        p1 = [face_landmarks.landmark[226].x, face_landmarks.landmark[226].y,
                              face_landmarks.landmark[226].z]
                        p2 = [face_landmarks.landmark[446].x, face_landmarks.landmark[446].y,
                              face_landmarks.landmark[446].z]
                        p3 = [face_landmarks.landmark[2].x, face_landmarks.landmark[2].y,
                              face_landmarks.landmark[2].z]
                        source_points = np.array([p1, p2, p3])
                        # print(source_points)
                        R, t, s = est_similarity_trans(source_points)
                        for id, lm in enumerate(face_landmarks.landmark):
                            keypoints[id, :] = [lm.x, lm.y, lm.z]

                        transformed_keypoints = similarity_trans(keypoints, R, t, s)

                        for id, lm in enumerate(face_landmarks.landmark):
                            keypoints_frame[0, i] = transformed_keypoints[id, 0]
                            i += 1
                            keypoints_frame[0, i] = transformed_keypoints[id, 1]
                            i += 1

                            if i == IN_DIM:
                                ## complete collection of 2 secs keypoints, pass to AI to do pain recognition
                                keypoints_frame = keypoints_frame.astype(np.float32)
                                tensor1 = torch.from_numpy(keypoints_frame)
                                inputs = Variable(tensor1)
                                out = net(inputs)
                                max, index = torch.max(out, dim=1)
                                if index[0] == 0:
                                    text = "Pain"
                                elif index[0] == 1:
                                    text = "Mild pain"
                                else:
                                    text = "No pain"

                                i = 0  # count


                       # mp_drawing.draw_landmarks(
                       #     image=image,
                       #     landmark_list=face_landmarks,
                        #    connections=mp_face_mesh.FACEMESH_TESSELATION,
                        #    landmark_drawing_spec=None,
                         #   connection_drawing_spec=mp_drawing_styles
                         #   .get_default_face_mesh_tesselation_style())
                #                mp_drawing.draw_landmarks(
                #                    image=image,
                #                    landmark_list=face_landmarks,
                #                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                #                    landmark_drawing_spec=None,
                #                    connection_drawing_spec=mp_drawing_styles
                #                    .get_default_face_mesh_contours_style())
                #                mp_drawing.draw_landmarks(
                #                    image=image,
                #                    landmark_list=face_landmarks,
                #                    connections=mp_face_mesh.FACEMESH_IRISES,
                #                    landmark_drawing_spec=None,
                #                    connection_drawing_spec=mp_drawing_styles
                #                    .get_default_face_mesh_iris_connections_style())
                # Flip the image horizontally for a selfie-view display.
                else:
                    text = 'No face detected'
                # image = cv2.flip(image,
                # 1)  # flip the image first before print the text, otherwise the text will be flipped also

                #cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), cv2.LINE_4)
                #cv2.namedWindow("AI pain detection - NYP", cv2.WINDOW_NORMAL)
                #cv2.setWindowProperty('AI pain detection - NYP', cv2.WND_PROP_TOPMOST, 1)
                #cv2.setWindowProperty('AI pain detection - NYP', cv2.WINDOW_FULLSCREEN, cv2.WND_PROP_TOPMOST)

                #cv2.imshow('AI pain detection - NYP', image)

                #def update_frame(image):
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                img_tk = ImageTk.PhotoImage(image=img)
                video_label.config(image=img_tk)
                video_label.image = img_tk  # Keep a reference to avoid garbage collection
                #time.sleep(1000)
                #    root.after(10, update_frame)  # Update every 10 milliseconds (approx. 30 FPS)
                #update_frame(image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

        ##########
        """""""""
        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                video_label.config(image=img_tk)
                video_label.image = img_tk  # Keep a reference to avoid garbage collection
            root.after(10, update_frame)  # Update every 10 milliseconds (approx. 30 FPS)

        update_frame()
        """


def show_live_video():
    cap = cv2.VideoCapture(0)  # 0 for the default webcam, you can change it if needed

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            video_label.config(image=img_tk)
            video_label.image = img_tk  # Keep a reference to avoid garbage collection
            #if cv2.waitKey(5) & 0xFF == 27:
            #    break
        root.after(10, update_frame)  # Update every 10 milliseconds (approx. 30 FPS)

    update_frame()

# Create the main tkinter window
root = tk.Tk()
root.title("AI pain detection")

# Maximize the window
root.state('zoomed')

# Create the first button to open the dialog box
button1 = tk.Button(root, text="open video files", command=open_dialog_box)
button1.pack(pady=10)

# Create the second button to show live webcam video
button2 = tk.Button(root, text="Live detection", command=show_live_video)
button2.pack(pady=10)

# Create a label to display the live video
video_label = tk.Label(root)
video_label.pack(pady=10)


# Start the main event loop
root.mainloop()
