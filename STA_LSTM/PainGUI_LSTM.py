import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import cv2
import numpy as np

import mediapipe as mp
from transform3d_scal import est_similarity_trans, similarity_trans
import torch
from torch.autograd import Variable

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, \
    QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

base = 2
IN_DIM = int(956 * 60 / base)  # always use 60 / base number to get the final datapoints, e.g. 60 / base_2 = 30
SEQUENCE_LENGTH = int(60 / base)  # 2

LSTM_IN_DIM = int(IN_DIM / SEQUENCE_LENGTH)
LSTM_HIDDEN_DIM = 64  # 32 #64

OUT_DIM = 3

class_labels = ['Severe Pain', 'Mild pain', 'No pain']

net = torch.load('C:\\projects\\STALSTM\\models\\manual_label_all_e60.pth')

net.eval()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
keypoints = np.zeros(shape=(478, 3))
# keypoints_frame = np.zeros(shape=(1, IN_DIM))
i = 0  # count of feature vector


class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create a figure and add a subplot for the histogram
        self.figure = Figure()
        self.hist_subplot = self.figure.add_subplot(111)

        # Create a canvas to display the histogram plot
        self.canvas = FigureCanvas(self.figure)

        # Create a layout to add the canvas
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.init_painscore_list()
        self.setLayout(layout)
        self.init_painscore_list()

    def init_painscore_list(self):
        duration_monitor = 10  # minutes
        self.pain_or_no_pain_list = [" "] * (duration_monitor * 60)

    def update_histogram(self):
        # Count the occurrences of "Pain" and "No pain"
        pain_count = self.pain_or_no_pain_list.count("Severe Pain")
        mild_pain_count = self.pain_or_no_pain_list.count("Mild pain")
        no_pain_count = self.pain_or_no_pain_list.count("No pain")
        total = pain_count + no_pain_count + mild_pain_count
        if not total == 0:
            perc_pain = pain_count * 100 / total
            perc_no_pain = no_pain_count * 100 / total
            perc_mild_pain = mild_pain_count * 100 / total
        else:
            perc_pain = 0
            perc_no_pain = 0
            perc_mild_pain = 0

        # Create labels and values for the histogram
        labels = ["Severe Pain%", "Mild pain%", "No pain%"]
        values = [perc_pain, perc_mild_pain, perc_no_pain]
        colors = ['orange', 'blue', 'green']
        # Clear the previous histogram and plot the new one
        self.hist_subplot.clear()
        self.hist_subplot.bar(labels, values, color=colors)
        self.hist_subplot.set_ylim(0, 100)

        # Refresh the canvas to update the plot
        self.canvas.draw()

    # def custom_processing(frame):
    # Add your custom video processing code here
    # For example, you can apply filters, object detection, or other image manipulations
    # Modify the frame as needed and return the processed frame
    """""""""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    label = " "
    if results.detections:
        for detection in results.detections:
            # Retrieve bounding box coordinates`
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            xmax = int((bbox.xmin + bbox.width) * w)
            ymax = int((bbox.ymin + bbox.height) * h)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            roi_color = frame[ymin:ymax, xmin: xmax]
            # cv2.imshow('ROI',roi_color)
            roi_color = cv2.resize(roi_color, (224, 224), interpolation=cv2.INTER_AREA)

            # Get image ready for prediction
            roi = roi_color.astype('float') / 255.0  # Scale
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 224, 224, 3)

            preds = emotion_model.predict(roi)[0]  # Yields one hot encoded result for 7 classes
            label = class_labels[preds.argmax()]  # Find the label

            label_position = (xmin, ymin)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # print(label)

    return label
    """


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STA-LSTM pain detection - NYP")
        # self.showMaximized()
        self.setWindowState(Qt.WindowMaximized)

        # self.showFullScreen()
        # self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel()
        # self.video_label.setFixedSize(1400, 1200)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)

        self.play_button = QPushButton("Start Live detection")
        self.play_button.clicked.connect(self.live_detection)

        self.stop_button = QPushButton("open file for processing")
        self.stop_button.clicked.connect(self.detection_file)

        self.text_label = QLabel("Pain histogram for past 10 minutes")
        self.text_label.setFixedSize(400, 20)
        self.text_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Set size policy to Fixed

        self.histogram_widget = HistogramWidget()
        self.histogram_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.logo_label = QLabel(self)
        logo_pixmap = QPixmap("painGUIlogo.png")  # Replace "logo.png" with the actual logo file name
        self.logo_label.setPixmap(logo_pixmap)
        self.logo_label.setFixedSize(1100, 120)
        # self.logo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.logo_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Set size policy to Fixed

        # Add margins around the logo label (optional)
        # self.logo_label.setContentsMargins(10, 10, 10, 10)
        self.keypoints_buffer = []
        # Add the logo label to the main layout

        layout1 = QHBoxLayout()
        layout1.addWidget(self.video_label)

        layout3 = QVBoxLayout()
        layout3.addWidget(self.logo_label, alignment=Qt.AlignHCenter)
        layout3.addWidget(self.text_label, alignment=Qt.AlignHCenter)
        layout3.addWidget(self.histogram_widget)

        layout1.addLayout(layout3)
        layout2 = QHBoxLayout()
        layout2.addWidget(self.play_button)
        layout2.addWidget(self.stop_button)
        mainlayout = QVBoxLayout()

        mainlayout.addLayout(layout1)
        mainlayout.addLayout(layout2)

        central_widget = QWidget()
        central_widget.setLayout(mainlayout)
        self.setCentralWidget(central_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False

    def live_detection(self):
        if not self.is_playing:
            self.is_playing = True
            self.video_capture = cv2.VideoCapture(0)
            self.histogram_widget.init_painscore_list()
            self.histogram_widget.update_histogram()
            self.play_button.setText("Stop live detection")
            self.timer.start(30)  # Set the playback speed (in milliseconds)
        else:
            self.is_playing = False
            self.play_button.setText("Start live detection")
            self.timer.stop()
            self.video_capture.release()

    def detection_file(self):

        if not self.is_playing:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                                       "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
            self.video_capture = cv2.VideoCapture(file_name)
            self.histogram_widget.init_painscore_list()
            self.histogram_widget.update_histogram()
            self.is_playing = True
            self.stop_button.setText("Stop detection from video file")
            self.timer.start(30)  # Set the playback speed (in milliseconds)
        else:
            self.is_playing = False
            self.stop_button.setText("Start detection from video file")
            self.timer.stop()
            self.video_capture.release()

    def update_frame(self):

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # refine landmarks using attention
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            try:
                if self.is_playing and self.video_capture.isOpened():

                    ret, frame = self.video_capture.read()

                    if not ret:
                        self.timer.stop()  # Stop the timer since the video has ended
                        self.is_playing = False
                        self.stop_button.setText("Start detection from video file")
                        self.video_capture.release()  # Release the video capture resources
                        return

                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:

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
                            self.keypoints_buffer = np.append(self.keypoints_buffer,
                                                              transformed_keypoints[:, :2].flatten())

                            if len(self.keypoints_buffer) >= 28680:
                                keypoints_frame = self.keypoints_buffer.astype(np.float32)
                                keypoints_frame = np.expand_dims(keypoints_frame, axis=0)
                                tensor1 = torch.from_numpy(keypoints_frame)
                                inputs = Variable(tensor1)
                                out = net(inputs)
                                label = class_labels[out.argmax()]
                                # print(label)
                                self.histogram_widget.pain_or_no_pain_list.pop(0)
                                self.histogram_widget.pain_or_no_pain_list.append(label)
                                self.histogram_widget.update_histogram()
                                self.keypoints_buffer = []
                        # frame.flags.writeable = True
                        y_position = int((face_landmarks.landmark[10].y - (
                                0.2 * (face_landmarks.landmark[152].y - face_landmarks.landmark[10].y))) *
                                         frame.shape[0])
                        if y_position <= 0:
                            y_position = 50

                        label_position = (int(face_landmarks.landmark[226].x * frame.shape[1]), y_position)
                        cv2.putText(frame,
                                    self.histogram_widget.pain_or_no_pain_list[-1],
                                    label_position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0),  # B, G, R
                                    3,
                                    cv2.LINE_4)
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1,
                                                                         circle_radius=1))

                    """""""""
                        connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                         circle_radius=1)

                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=connection_drawing_spec )

                        """

                    h, w, ch = frame.shape
                    image = QImage(frame, w, h, ch * w, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(image))
                    self.timer.start(30)  # Set the playback speed (in milliseconds)
            except Exception as e:
                print("Error:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
