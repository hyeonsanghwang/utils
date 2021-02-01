import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import cv2
from visualization.common import draw_fps

DRAW_TYPE_LINE = 0
DRAW_TYPE_BAR = 1


def signal_to_frame(signal,
                    width=-1,
                    height=100,
                    frame=None,
                    draw_type=DRAW_TYPE_LINE,
                    thickness=3,
                    foreground=(0, 0, 255),
                    background=(0, 0, 0),
                    scale=None,
                    ret_scale=False,
                    padding=2):
    data = np.array(signal, np.float)
    length = data.shape[0]

    if frame is None:
        width = (length + padding * 2) if width == -1 else width
        frame = np.ones((height, width, 3), np.uint8) * np.array(background, np.uint8)
    else:
        height, width, _ = frame.shape
    padded_width, padded_height = width - padding * 2, height - padding * 2

    data_min, data_max = np.min(data), np.max(data)
    if scale is not None:
        data_min, data_max = np.min(scale), np.max(scale)
        data[data > data_max] = data_max
        data[data < data_min] = data_min

    term = data_max - data_min
    if term == 0:
        normed = np.zeros(data.shape, np.float)
    else:
        normed = (data - data_min) / term

    if draw_type == DRAW_TYPE_LINE:
        for i in range(length-1):
            sx = int(padding + (padded_width / length) * i)
            sy = int(height - padding - (padded_height * normed[i]))
            ex = int(padding + (padded_width / length) * (i + 1))
            ey = int(height - padding - (padded_height * normed[i + 1]))
            cv2.line(frame, (sx, sy), (ex, ey), foreground, thickness)
    elif draw_type == DRAW_TYPE_BAR:
        for i in range(length):
            sx = int(padding + (padded_width / length) * i)
            sy = int(height - padding - (padded_height * normed[i]))
            ex = int(padding + (padded_width / length) * (i + 1))
            ey = int(height - 1)
            cv2.rectangle(frame, (sx, sy), (ex, ey), foreground, -1)
    return (frame, (data_min, data_max)) if ret_scale else frame


def show_signal(name,
                signal,
                width=-1,
                height=100,
                frame=None,
                draw_type=DRAW_TYPE_LINE,
                thickness=3,
                foreground=(0, 0, 255),
                background=(0, 0, 0),
                scale=None,
                ret_scale=False,
                padding=2,
                fps_info=None):

    frame = signal_to_frame(signal, width, height, frame, draw_type, thickness, foreground, background, scale,
                            ret_scale, padding)
    if fps_info is not None:
        fps = fps_info[0] if fps_info else 0
        text = "%d fps" if len(fps_info) < 2 else fps_info[1]
        color = (0, 0, 255) if len(fps_info) < 3 else fps_info[2]
        draw_fps(frame, fps, text, color)

    cv2.imshow(name, frame)

