import cv2
import mediapipe as mp
import numpy as np
#from rigid_transform_3D import est_similarity_trans, similarity_trans
#from transform3d import est_similarity_trans, similarity_trans
from transform3d_scal import est_similarity_trans, similarity_trans
def keypoints_extract(painornot, nseconds):
    # this function take in videos from webcam, return a nx28080 array
    text = "3D normalization demo"

    train_data = np.empty(shape=(0, 28680))

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
    keypoints_frame = np.zeros(shape=(1, 28680)) # create a zeros numpy array to store keypoints corrdinates for 30 frames 468x2x30 = 28080
    keypoints = np.zeros(shape=(478,3))
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # refine landmarks using attention
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
                    p1= [face_landmarks.landmark[226].x,face_landmarks.landmark[226].y,face_landmarks.landmark[226].z]
                    p2 = [face_landmarks.landmark[446].x, face_landmarks.landmark[446].y,
                          face_landmarks.landmark[446].z]
                    p3 = [face_landmarks.landmark[2].x, face_landmarks.landmark[2].y,
                          face_landmarks.landmark[2].z]
                    source_points = np.array([p1, p2, p3])
                    #print(source_points)
                    R, t ,S = est_similarity_trans(source_points)
                    #R, t = est_similarity_trans(source_points)
                    for id, lm in enumerate(face_landmarks.landmark):
                        #transformed_keypoints = similarity_trans(np.array([lm.x,lm.y,lm.z]), R, t, S)
                        keypoints[id,:]=[lm.x,lm.y,lm.z]

                    transformed_keypoints = similarity_trans(keypoints,R,t,S)
                    #print(transformed_keypoints)
                    for id, lm in enumerate(face_landmarks.landmark):
                        face_landmarks.landmark[id].x = transformed_keypoints[id,0]
                        face_landmarks.landmark[id].y = transformed_keypoints[id, 1]
                        face_landmarks.landmark[id].z = transformed_keypoints[id, 2]
                        keypoints_frame[0, i] = transformed_keypoints[id, 0]
                        i += 1
                        keypoints_frame[0, i] = transformed_keypoints[id, 1]
                        i += 1
                        if i == 28680:
                            train_data = np.vstack([train_data, keypoints_frame])
                            j += 1
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
            cv2.putText(image, painornot,(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2, cv2.LINE_4)
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
#keypoints_extract("pain", 1000)
#data1 = keypoints_extract("pain",3)
#print(data1)