import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import cv2
from visualization.common import draw_fps
from processing.normalize import min_max_normalize

DRAW_TYPE_LINE = 0
DRAW_TYPE_BAR = 1


def signal_to_frame(signal,                     # Signal
                    width=-1,                   # Frame info (width)
                    height=100,                 # Frame info (height)
                    frame=None,                 # Frame info (to draw)
                    padding=2,                  # Frame info (padding)
                    draw_type=DRAW_TYPE_LINE,   # Line info (line type)
                    thickness=3,                # Line info (thickness)
                    foreground=(0, 0, 255),     # Color info (foreground)
                    background=(0, 0, 0),       # Color info (background)
                    scale=None,                 # Scale info (target scale of value)
                    ret_scale=False,            # Scale info (return scale)
                    flip=False,                 # Flip
                    circle_indexes=[],          # Circle info (indexes)
                    circle_color=(0, 0, 255),   # Circle info (color)
                    circle_radius=5):           # Circle info (radius)

    # Validate data
    try:
        data = np.array(signal, np.float)
        data_len = data.shape[0]
        if len(data.shape) > 1:
            print("[ERROR] signal_to_frame: Signal must be one-dimensional.")
            return (None, None) if ret_scale else None
        if data_len < 2:
            print("[ERROR] signal_to_frame: The length of signal must be 2 or more.")
            return (None, None) if ret_scale else None
    except:
        print("[ERROR] signal_to_frame: Unknown data type (list or np.array).")
        return (None, None) if ret_scale else None

    # Set frame
    if frame is None:
        width = (data_len + padding * 2) if width == -1 else width
        frame = np.ones((height, width, 3), np.uint8) * np.array(background, np.uint8)
    else:
        height, width, _ = frame.shape
    pad_w, pad_h = width - padding * 2, height - padding * 2

    # Normalize data
    data = (data * -1) if flip else data
    normed, (data_min, data_max) = min_max_normalize(data, scale, axis=-1, ret_min_max=True)

    # Draw data
    if draw_type == DRAW_TYPE_LINE:
        for i in range(data_len - 1):
            sx = int(padding + (pad_w / data_len) * i)
            sy = int(height - padding - (pad_h * normed[i]))
            ex = int(padding + (pad_w / data_len) * (i + 1))
            ey = int(height - padding - (pad_h * normed[i + 1]))
            cv2.line(frame, (sx, sy), (ex, ey), foreground, thickness)
            if i in circle_indexes:
                cv2.circle(frame, ((sx + ex) // 2, (sy + ey) // 2), circle_radius, circle_color, -1)
    elif draw_type == DRAW_TYPE_BAR:
        for i in range(data_len):
            sx = int(padding + (pad_w / data_len) * i)
            sy = int(height - padding - (pad_h * normed[i]))
            ex = int(padding + (pad_w / data_len) * (i + 1))
            ey = int(height - 1)
            cv2.rectangle(frame, (sx, sy), (ex, ey), foreground, -1)

    return (frame, (data_min, data_max)) if ret_scale else frame


def show_signal(name,                       # Window name
                signal,                     # Signal
                width=-1,                   # Frame info (width)
                height=100,                 # Frame info (height)
                frame=None,                 # Frame info (to draw)
                padding=2,                  # Frame info (padding)
                draw_type=DRAW_TYPE_LINE,   # Line info (line type)
                thickness=3,                # Line info (thickness)
                foreground=(0, 0, 255),     # Color info (foreground)
                background=(0, 0, 0),       # Color info (background)
                scale=None,                 # Scale info (target scale of value)
                ret_scale=False,            # Scale info (return scale)
                flip=False,                 # Flip
                circle_indexes=[],          # Circle info (indexes)
                circle_color=(0, 0, 255),   # Circle info (color)
                circle_radius=5,            # Circle info (radius)
                fps_text=None,              # Fps info (string)
                fps_color=(0, 255, 255),    # Fps info (color)
                fps_location=(5, 20)):      # Fps info (location)

    ret = signal_to_frame(signal=signal,
                          width=width,
                          height=height,
                          frame=frame,
                          padding=padding,
                          draw_type=draw_type,
                          thickness=thickness,
                          foreground=foreground,
                          background=background,
                          scale=scale,
                          ret_scale=ret_scale,
                          flip=flip,
                          circle_indexes=circle_indexes,
                          circle_color=circle_color,
                          circle_radius=circle_radius)
    if ret_scale:
        frame, scale = ret
    else:
        frame = ret

    if fps_text is not None:
        frame = draw_fps(frame, fps_text, fps_color, fps_location)

    cv2.imshow(name, frame)
    return (frame, scale) if ret_scale else frame


sin_signals = []
sin_index = 0
def show_sin_signals(name="BPMs",
                     fps=30,
                     duration=15,
                     window_size=None,
                     bpms=(10, 15, 20, 25, 30, 35, 40),
                     signal_width=500,
                     signal_height=100,
                     frame_margin=5,
                     init_data=False):
    global sin_signals, sin_index
    if init_data:
        sin_signals = []
        sin_index = 0

    window_size = fps * duration if window_size is None else window_size

    # Create bpm frame
    n_bpms = len(bpms)
    bpm_frame = np.zeros((frame_margin * (n_bpms + 1) + signal_height * n_bpms, signal_width, 3), np.uint8)

    # Draw sin signals
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

