import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import os
import numpy as np
import cv2
from time import perf_counter


class VideoStream:
    FORMAT_WEBCAM = 0
    FORMAT_VIDEO = 1
    FORMAT_NUMPY = 2

    def __init__(self, src, fps=None, width=640, height=480, time_window=0):
        self.src = src
        self.fps = fps
        self.width = width
        self.height = height
        self.window = 2 if time_window == 0 else time_window * fps

        self.status = None
        self.stream = None
        self.curr_index = 0
        self.time_stamp = []

        if isinstance(self.src, int):
            self.status = self.FORMAT_WEBCAM
            self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            self.fps = 30 if self.fps is None else self.fps
            self.set(cv2.CAP_PROP_FPS, self.fps)
            self.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        elif isinstance(self.src, str):
            extension = os.path.splitext(self.src)[1]
            if extension == '.npy':
                self.status = self.FORMAT_NUMPY
                self.stream = np.load(self.src)
                self.fps = 30 if self.fps is None else self.fps
            else:
                self.status = self.FORMAT_VIDEO
                self.stream = cv2.VideoCapture(self.src)
                self.fps = self.get(cv2.CAP_PROP_FPS) if self.fps is None else self.fps
                self.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            print("Unknown format: src")
            return

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FPS:
            self.fps = value
            if self.status == self.FORMAT_WEBCAM and self.stream is not None:
                self.stream.set(prop, value)
        if self.status == self.FORMAT_WEBCAM or self.status == self.FORMAT_VIDEO:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self.width = value
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = value
            return self.stream.set(prop, value)


    def get(self, prop):
        if self.status == self.FORMAT_WEBCAM or self.status == self.FORMAT_VIDEO:
            return self.stream.get(prop)

    def read(self):
        self.time_stamp.append(perf_counter())
        if len(self.time_stamp) > self.window:
            del self.time_stamp[0]

        if self.status == self.FORMAT_WEBCAM or self.status == self.FORMAT_VIDEO:
            ret, frame = self.stream.read()
        else:
            try:
                ret, frame = True, self.stream[self.curr_index]
            except Exception as e:
                ret, frame = False, None
        self.curr_index += 1
        return ret, frame

    def delay(self):
        if self.status == self.FORMAT_WEBCAM:
            return 1
        else:
            perf_time = perf_counter() - self.time_stamp[0]
            return max(int(((float(len(self.time_stamp)) / self.fps) - perf_time) * 1000), 1)

    def get_fps(self):
        if len(self.time_stamp) > 1:
            perf_time = (self.time_stamp[-1] - self.time_stamp[0]) / (len(self.time_stamp) - 1)
            return 1.0 / perf_time
        else:
            return 0

    def get_fps_without_delay(self):
        if len(self.time_stamp) > 0:
            perf_time = perf_counter() - self.time_stamp[-1]
            return 1.0 / perf_time
        else:
            return 0

    def release(self):
        if self.status == self.FORMAT_WEBCAM or self.status == self.FORMAT_VIDEO:
            self.stream.release()
