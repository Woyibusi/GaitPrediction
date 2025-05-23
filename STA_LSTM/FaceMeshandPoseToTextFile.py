import time
import os
import cv2
import mediapipe as mp
import datetime
import pytz
import numpy as np
import threading


def faceposeDetector(path):
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            file_dir = os.path.join(path, file)
            print(file_dir)
            cap = cv2.VideoCapture(file_dir)
            # ## To save video output
            # if int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) > 1920:  ## video higher than Full HD
            #     FrameSize = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
            # else:
            #     FrameSize = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # length = int(cap.get(cv2.CAP_PROP_FPS))
            # print(FrameSize, length)
            # out = cv2.VideoWriter(file_dir.split(".")[0] + "-output.avi", cv2.VideoWriter_fourcc(*'MJPG'), length,
            #                       FrameSize)
            # ##
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
            while(cap.isOpened()):
                success, img = cap.read()
                if success == True:
                    frame += 1
                    backgnd_img = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results_fm = faceMesh.process(imgRGB)
                    if results_fm.multi_face_landmarks:
                        face_array = []
                        for faceLms in results_fm.multi_face_landmarks:
                            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                                  drawSpec,drawSpec)
                            for id,lm in enumerate(faceLms.landmark):
                                ih, iw, ic = img.shape
                                x, y, z = int(lm.x*iw), int(lm.y*ih), lm.z
                                face_list.append([frame, id, x, y, z])
                                face_array.append([x, y, z])
                                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), 1)
                                cv2.circle(backgnd_img, (int(x), int(y)), 3, (255, 255, 255), 1)
                            frames_arr.append(face_array)
                    # to_display = np.hstack((img, backgnd_img))
                    # if to_display.shape[0] > 1920:
                    #     to_display = cv2.resize(to_display, (FrameSize[0], FrameSize[1]))
                    # out.write(to_display)
                    results_p = pose.process(imgRGB)
                    # print(results.pose_landmarks)
                    if results_p.pose_landmarks:
                        mpDraw.draw_landmarks(img, results_p.pose_landmarks, mpPose.POSE_CONNECTIONS)
                        for id, lm in enumerate(results_p.pose_landmarks.landmark):
                            h, w, c = img.shape
                            cx, cy, z = int(lm.x * w), int(lm.y * h), lm.z
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                            pose_list.append([frame, id, cx, cy, z])
                else:
                    cap.release()
                    # out.release()
                    break

            ## save frames_arr as .npy file
            outfile = os.path.join(path, file.split('.')[0] + '.npy')
            np.save(outfile, frames_arr)

            ## save as text file
            fname = os.path.join(path, file.split(".")[0] + "-faciallandmarks.txt")
            textfile = open(fname, "w")
            for element in face_list:
                textfile.write((", ".join(str(x) for x in element)) + "\n")
            textfile.close()
            fname = os.path.join(path, file.split(".")[0] + "-pose.txt")

            textfile = open(fname, "w")
            for element in pose_list:
                textfile.write((", ".join(str(x) for x in element)) + "\n")
            textfile.close()

folder_path = r"C:\projects\STALSTM" ## insert folder path directory here
faceposeDetector(folder_path)