import cv2
import mediapipe as mp
import numpy as np
from transform_2d import est_similarity_trans, similarity_trans


def keypoints_extract(painornot, nseconds):
    # this function take in videos from webcam, return a nx28080 array
    text = "recording " + painornot + " video"

    train_data = np.empty(shape=(0, 28080))

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # For webcam or video input:
    # filename = r"C:\projects\Facial Landmark and Pose Estimation Project\AD_Snr1__20220518_Sp1_135548.mp4"


    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    #cap = cv2.VideoCapture(filename)
    cap = cv2.VideoCapture(0)

    evenodd = True # a variable to remember if the frame is even or odd frame
    i=0            # count
    j = 0
    keypoints_frame = np.zeros(shape=(1, 28080)) # create a zeros numpy array to store keypoints corrdinates for 30 frames 468x2x30 = 28080
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,  # refine landmarks using attention
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            #i += 1
            # only process every 2 frames #duth
            """ ""
            if evenodd:
                evenodd = False
            else:
                evenodd = True
                continue
            """
            #print(i)
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
                    x1= face_landmarks.landmark[10].x
                    y1= face_landmarks.landmark[10].y
                    x2 = face_landmarks.landmark[152].x
                    y2 = face_landmarks.landmark[152].y
                    T =est_similarity_trans(x1, y1, x2, y2)
                    for id, lm in enumerate(face_landmarks.landmark):
                        new_xy = similarity_trans(lm.x, lm.y, T)
                        keypoints_frame[0,i] = new_xy[0][0]
                        i +=1
                        keypoints_frame[0,i] = new_xy[0][1]
                        i +=1

                        if i == 28080:

                            train_data = np.vstack([train_data, keypoints_frame])
                            j +=1
                            i = 0  # once reach 28080, reset count

                            break


                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
            image = cv2.flip(image,1)  # flip the image first before print the text, otherwise the text will be flipped also
            cv2.putText(image,
                        text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), # B, G, R
                        2,
                        cv2.LINE_4)
            cv2.namedWindow("AI pain detection - NYP", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('AI pain detection - NYP', cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty('AI pain detection - NYP', cv2.WINDOW_FULLSCREEN, cv2.WND_PROP_TOPMOST)
            cv2.imshow('AI pain detection - NYP', image)
            if j == nseconds: ## only record nsecondsx2 seconds of video
                break
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        return train_data

#data1 = keypoints_extract("pain")
#print(data1)