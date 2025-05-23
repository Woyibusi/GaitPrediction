import cv2
import mediapipe as mp
import numpy as np
from transform_2d import est_similarity_trans, similarity_trans
from keypoints_extract import keypoints_extract
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

base = 2
IN_DIM = int(936 * 60 / base)  # always use 60 / base number to get the final datapoints, e.g. 60 / base_2 = 30
SEQUENCE_LENGTH = int(60 / base)  # 2

LSTM_IN_DIM = int(IN_DIM / SEQUENCE_LENGTH)
LSTM_HIDDEN_DIM = 64  # 32 #64

OUT_DIM = 3
text = "No Pain"

LEARNING_RATE = 0.05  # learning rate
WEIGHT_DECAY = 1e-6  # not used as the number of epoches is only 8 or 10
EPOCHES = 20 ## 10 epoches can reach 100%
USE_GPU = False

## if the training model is saved in the folder, then use it, otherwise load a pretrained model, then apply .transfer learning
username = input("please key in your name")
try:
    net =torch.load("C:\\projects\\STALSTM\\models\\" + username + "_personalized_train.pth")
    print("welcome back! "+ username)
    training_required = False
except:
    print("your profile is not found in the system, a personalized calibration is required")
    net = torch.load('C:\\projects\\STALSTM\\models\\SGH_26to100_b2_e100.pth')
    training_required = True




#net = torch.load('C:\\projects\\STALSTM\\models\\SGH_26to100_b2_e100.pth')
#net = torch.load('C:\\projects\\STALSTM\\models\\Du Tiehua_personalized_train.pth')

""""
transfer learning
record a video of pain, then record a video without pain
"""""""""
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 学习率根据训练的次数进行调整
adjust_lr = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[i * 10 for i in range(EPOCHES // 10)],
                                           gamma=0.5)

# 定义训练损失函数&测试误差函数
# loss_criterion = nn.SmoothL1Loss()
loss_criterion = nn.MSELoss()
error_criterion = nn.L1Loss()  # MSELoss()

### freeze all layers, except the last one


for name,param in net.named_parameters():
    if name not in ['T_A.weight', 'T_A.bias','layer_out.weight']:
        param.requires_grad = False





def train(inputs, groundtruths):
    net.train()
    loss_list = []
    inputs = inputs
    groundtruths = groundtruths
    optimizer.zero_grad()
    # 获得网络输出结果
    out = net(inputs)
    # 根据真值计算损失函数的值
    loss = loss_criterion(out, groundtruths)
    out = out.detach().numpy()
    groundtruths = groundtruths.detach().numpy()
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(groundtruths, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 通过优化器优化网络
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

    return loss_list, acc


#user_option = input('Enter 0 for live pain level estimation, enter 1 for personalized calibration')

#### prepare training data
# train_data -- nx28080 array
# train_ground truth nx3 array 100 pain, 001 no pain

if training_required:
    msg_display("pain", 3)
    train_data_pain = keypoints_extract('Pain', 8)
    msg_display("No pain",3)
    train_data_nopain = keypoints_extract('No pain', 8)

    train_data = np.vstack([train_data_pain, train_data_nopain])
    train_data = train_data.astype(np.float32)
    train_data = torch.from_numpy(train_data)
    inputs = Variable(train_data)
    ground_truth_pain = np.zeros(shape=(8, 3))
    ground_truth_pain[:, 0] = 1
    ground_truth_nopain = np.zeros(shape=(8, 3))
    ground_truth_nopain[:, -1] = 1
    train_groundtruth = np.vstack([ground_truth_pain, ground_truth_nopain])
    train_groundtruth = train_groundtruth.astype(np.float32)
    train_groundtruth = torch.from_numpy(train_groundtruth)
    train_data = Variable(train_data)
    train_groundtruth = Variable(train_groundtruth)
    #msg_display("pain", 3)
    ## start training
    train_start = time.time()
    loss_recorder = []

    print('starting training... ')

    for epoch in range(EPOCHES):
        print(epoch)

        loss_list, acc = train(train_data, train_groundtruth)

        loss_recorder.append(np.mean(loss_list))

        acc = acc * 100.0
        adjust_lr.step()

        print('\nepoch = %d \nloss = %.5f, accuracy = %2.5f' % (epoch + 1, np.mean(loss_list), acc))
    """
        if (epoch % 5 == 0):
            test_start = time.time()
            acc, average_error = test()

            acc = acc * 100.0
            print('Loss = %.5f, Test accuracy is = %2.5f' % (average_error, acc.numpy()))
    """

    print('training time = {}s'.format(int((time.time() - train_start))))



    torch.save(net, "C:\\projects\\STALSTM\\models\\" + username + "_personalized_train.pth")
    text = "System calibration completed"



#######


net.eval()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam or video input:

#filename = r"C:\projects\Facial Landmark and Pose Estimation Project\AD_Snr1__20220518_Sp1_135548.mp4"
# filename = "AD_Snr1__20220518_Sp1_135548.mp4"
# cap = cv2.VideoCapture(filename)
cap = cv2.VideoCapture(0)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
evenodd = True  # a variable to remember if the frame is even or odd frame
i = 0  # count of feature vector
keypoints_frame = np.zeros(shape=(1, 28080))  # create a zeros numpy array to store keypoints corrdinates for 30 frames 468x2x30 = 28080

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,  # refine landmarks using attention
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # i += 1
        # only process every 2 frames #duth
        """
        if evenodd:
            evenodd = False
        else:
            evenodd = True
            continue
        """
        # print(i)
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
                x1 = face_landmarks.landmark[10].x
                y1 = face_landmarks.landmark[10].y
                x2 = face_landmarks.landmark[152].x
                y2 = face_landmarks.landmark[152].y
                T = est_similarity_trans(x1, y1, x2, y2)
                for id, lm in enumerate(face_landmarks.landmark):
                    new_xy = similarity_trans(lm.x, lm.y, T)
                    keypoints_frame[0, i] = new_xy[0][0]
                    i += 1
                    keypoints_frame[0, i] = new_xy[0][1]
                    i += 1

                    if i == 28080:
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
                        # keypoints_frame = np.zeros(shape=(1, 28080)) # need not set to zero array, just reuse
                        break
                # print(lm.x)
                # print(lm.y )

                # for id, lm in enumerate(face_landmarks.landmark):
                # ih, iw, ic = img.shape
                # x, y, z = int(lm.x * iw), int(lm.y * ih), lm.z
                # frames_arr.append(lm.x)
                # frames_arr.append(lm.y)

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
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
            text ='No face detected'
        image = cv2.flip(image,
                         1)  # flip the image first before print the text, otherwise the text will be flipped also
        cv2.putText(image,
                    text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255),  # B, G, R
                    2,
                    cv2.LINE_4)
        cv2.namedWindow("AI pain detection - NYP", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('AI pain detection - NYP', cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty('AI pain detection - NYP', cv2.WINDOW_FULLSCREEN, cv2.WND_PROP_TOPMOST)

        cv2.imshow('AI pain detection - NYP', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
