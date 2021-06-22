import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2


def draw_fps(frame, fps, color=(0, 0, 255), location=(5, 20)):
    image = frame.copy()
    cv2.putText(image, fps, location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
