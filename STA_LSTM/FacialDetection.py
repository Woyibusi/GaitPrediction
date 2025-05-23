
import time
import os
import cv2
import mediapipe as mp
import datetime
import pytz
import numpy as np
import threading


def faceposeDetector(filename):
    if filename.endswith(".mp4"):
        cap = cv2.VideoCapture(filename)
        mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.3, min_tracking_confidence=0.2)
        mpPose = mp.solutions.pose
        pose = mpPose.Pose()
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        face_list = [["frame", "id", "x", "y", "z"]]
        pose_list = [["frame", "id", "x", "y", "z"]]
        frames_arr = []

        frame = 0
        i=0
        while(cap.isOpened()):
            success, img = cap.read()
            if success == True:
                frame += 2
                i +=1
                print(frame, i)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results_fm = faceMesh.process(imgRGB)
                if results_fm.multi_face_landmarks:
                    # face_array = []
                    for faceLms in results_fm.multi_face_landmarks:
                        mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                              drawSpec,drawSpec)
                        for id,lm in enumerate(faceLms.landmark):
                            ih, iw, ic = img.shape
                            x, y, z = int(lm.x*iw), int(lm.y*ih), lm.z
                            frames_arr.append(x)
                            frames_arr.append(y)
                        # frames_arr.append(face_array)
            else:
                cap.release()
                break
    return frames_arr
def est_similarity_trans(x1,y1,x2,y2):
    #x1,y1,x2,y2 are the corrdinates of facial keypoints 10 and 152 (using mediapipe)
    x1p,y1p,x2p,y2p=0.5,0.4,0.5,0.6 # new corrdiantes after similarity transformation within (0,1)
    A=np.array([[x1,-y1,1,0],[y1,x1,0,1],[x2,-y2,1,0],[y2,x2,0,1]])
    b=np.array([[x1p],[y1p],[x2p],[y2p]])
    T_matrix=np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.dot(np.transpose(A),b))
    return T_matrix

def similarity_trans(x,y,T):
    A = np.array([[x, -y, 1, 0], [y, x, 0, 1]])
    b=np.dot(A,T)
    #print(np.transpose(b))
    b1=np.dot(np.transpose(b),[[2160,0],[0,3840]])
    return b1


frames_arr = faceposeDetector("PD_Snr1_Unknown_20220719_Sp5_210313.mp4")
frames_arr1 = []
T = est_similarity_trans(frames_arr[20], frames_arr[21], frames_arr[304], frames_arr[305])
for i in range(0, len(frames_arr), 2):
    new_xy=similarity_trans(int(frames_arr[i]), int(frames_arr[i+1]), T)
    frames_arr1.append(new_xy[0][0])
    frames_arr1.append(new_xy[0][1])

print(len(frames_arr))