import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import cv2
from visualization.common import draw_fps
from processing.normalize import min_max_normalize

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
    if length == 0:
        data = np.array([0, 0, 0, 0, 0])

    if frame is None:
        width = (length + padding * 2) if width == -1 else width
        frame = np.ones((height, width, 3), np.uint8) * np.array(background, np.uint8)
    else:
        height, width, _ = frame.shape
    padded_width, padded_height = width - padding * 2, height - padding * 2

    normed, (data_min, data_max) = min_max_normalize(data, scale, axis=-1, ret_min_max=True)

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


sin_signals = []
sin_index = 0
def show_sin_signals(name="BPMs",
                     fps=30,
                     duration=15,
                     window_size=None,
                     bpms=(10, 15, 20, 25, 30, 35, 40),
                     signal_width=500,
                     signal_height=100,
                     frame_margin=5):

    window_size = fps * duration if window_size is None else window_size

    # Create bpm frame
    n_bpms = len(bpms)
    bpm_frame = np.zeros((frame_margin * (n_bpms + 1) + signal_height * n_bpms, signal_width, 3), np.uint8)

    # Draw sin signals
    global sin_index, sin_signals
    for i, bpm in enumerate(bpms):
        # Set sin signals
        sin_value = np.sin(sin_index * (2 * np.pi) / ((60.0 / bpm) * fps))
        if sin_index == 0:
            sin_signals.append([sin_value])
        else:
            sin_signals[i].append(sin_value)
            if len(sin_signals[i]) > window_size:
                del sin_signals[i][0]

        # Draw signals
        target_frame = bpm_frame[frame_margin * (i + 1) + signal_height * i: frame_margin * (i + 1) + signal_height * (i + 1),
                       ...]
        target_frame[...] = signal_to_frame(sin_signals[i], width=signal_width, height=signal_height, padding=0)
        cv2.putText(target_frame, '%02d bpm' % bpm, (signal_width-70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # increase sin index
    sin_index += 1

    # Show frame
    cv2.imshow(name, bpm_frame)

