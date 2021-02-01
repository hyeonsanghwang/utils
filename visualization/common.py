import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2


def draw_fps(frame, fps, text="%d fps", color=(0, 0, 255)):
    image = frame.copy()
    cv2.putText(image, text % fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
