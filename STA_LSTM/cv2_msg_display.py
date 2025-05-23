import time

import cv2
import numpy as np
def msg_display(painornot, nseconds):
    # this function take in videos from webcam, return a nx28080 array
    text = "Please get ready for " + painornot + " video recording"
    blank_image = np.ones((500, 1000, 3), np.uint8)


    for i in range(nseconds):
        blank_image = np.ones((500, 1000, 3), np.uint8)
        cv2.putText(blank_image,
                    text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255),  # B, G, R
                    2,
                    cv2.LINE_4)

        cv2.putText(blank_image,
                    str(nseconds-i)+" seconds",
                    (480, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255),  # B, G, R
                    2,
                    cv2.LINE_4)


        cv2.namedWindow("AI pain detection - NYP", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('AI pain detection - NYP', cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty('AI pain detection - NYP', cv2.WINDOW_FULLSCREEN, cv2.WND_PROP_TOPMOST)
        cv2.imshow('AI pain detection - NYP', blank_image)
        time.sleep(1)


        if cv2.waitKey(5) & 0xFF == 27:
            break



