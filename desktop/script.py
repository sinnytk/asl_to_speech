import sys
import cv2
import os
import mediapipe as mp

from PySide2.QtCore import Property, QObject, QTimer, Signal, Slot

from PySide2.QtGui import QGuiApplication, QImage
from PySide2.QtMultimedia import QAbstractVideoSurface, QVideoFrame, QVideoSurfaceFormat
from PySide2.QtQml import QQmlApplicationEngine, QQmlDebuggingEnabler 

from model import load_model, make_inference
class AbstractStreamAdapter(QObject):
    signal_video_surface_changed = Signal(QAbstractVideoSurface)

    def __init__(self):
        QObject.__init__(self)
        self._video_surface =  None
        self._surface_format = None
        self._timer = QTimer()
        self._timer.timeout.connect(self.get_frame)

        self._opencv_feed = None
        self.counter = 0

    def cv_image_to_q(self, cv_image):
        height, width, _ = cv_image.shape
        bytes_per_line = 4 * width
        q_img = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB32)
        return q_img

    def start_surface(self):
        frame = self._opencv_feed.get_frame()
        q_img = self.cv_image_to_q(frame)
        self._surface_format = QVideoSurfaceFormat(q_img.size(), QVideoFrame.pixelFormatFromImageFormat(q_img.format()))
        if not self._video_surface.isFormatSupported(self._surface_format):
            self._surface_format = self._video_surface.nearestFormat(self._surface_format)
        return self._video_surface.start(self._surface_format)

    @Slot()
    def get_frame(self):
        frame = self._opencv_feed.get_frame() 
        q_img = self.cv_image_to_q(frame)
        a = QVideoFrame(q_img)
        self._video_surface.present(a)

    @Property(QAbstractVideoSurface, notify=signal_video_surface_changed)
    def videoSurface(self):
        return self._video_surface

    @videoSurface.setter
    def videoSurface(self, video_surface):
        if self._video_surface == video_surface: 
            return
        self._video_surface = video_surface
        self.signal_video_surface_changed.emit(self._video_surface);
    
    def start(self, opencv_feed):
        self._opencv_feed = opencv_feed
        if not self.start_surface():
            return False
        self._timer.start(8)
        return True

class WebcamStream():
    hands = mp.solutions.hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    # change path to model before running
    model = load_model('../model/fully_connected_veri_good.pt')
    def __init__(self):
        self.cap = None
        self.open_camera()

    def get_frame(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = self.bound_hand(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        return frame

    def bound_hand(self, frame):
        h, w, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(framergb)

        hand_landmarks = result.multi_hand_landmarks
        handedness = result.multi_handedness

        if hand_landmarks and handedness[0].classification[0].label == 'Right':
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
            cv2.rectangle(frame, (x_min-25, y_min-25), (x_max+25, y_max+25), (0, 255, 0), 2)
            hand_frame = frame[y_min-25:y_max+25, x_min-25:x_max+25]
            inference = make_inference(self.model, hand_frame)
            print(inference)
        return frame

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)

    def close_camera(self):
        self.cap.release()

    def is_feed_open(self):
        return self.cap.isOpened()

class GUIBackend(QObject):
    statusChanged = Signal(str)
    feedToggled = Signal(bool)

    def __init__(self, stream_adapter):
        QObject.__init__(self)
        self._stream_adapter = stream_adapter
        self._status = "Initialized"
        self.feed = None
        self._is_feed_open = False
        
    @Property(bool, notify=feedToggled)
    def is_feed_open(self):
        return self._is_feed_open

    @Property(str, notify=statusChanged)
    def status(self):
        return self._status    

    @status.setter
    def status(self, newStatus):
        if self._status == newStatus:
            return
        self._status = newStatus
        self.statusChanged.emit(self._status)
    
    @is_feed_open.setter
    def is_feed_open(self, toggle):
        if self._is_feed_open == toggle:
            return
        self._is_feed_open = toggle
        self.feedToggled.emit(self._is_feed_open)

    @Slot()
    def start_webcam_feed(self):
        print("Starting webcam")
        if not self.feed:
            print("Feed doesn't exist, creating new one")
            self.feed = WebcamStream()
            if not self._stream_adapter.start(self.feed):
                self.status = "Something went wrong while parsing the stream, try again!"
                self.feed.close_camera()

        if self.feed.is_feed_open():
            print("Camera opened!")
            self.status = "Camera opened!"
            self.is_feed_open = True
        else:
            print("Camera could not be opened!")
            self.status = "Camera could not be opened!"
            self.is_feed_open = False


os.environ["QT_QUICK_CONTROLS_STYLE"] = "Material"
QQmlDebuggingEnabler()
app = QGuiApplication(sys.argv)
engine = QQmlApplicationEngine()
stream_adapter = AbstractStreamAdapter()
backend = GUIBackend(stream_adapter)
engine.rootContext().setContextProperty("GUIBackend", backend)
engine.rootContext().setContextProperty("AbstractStreamAdapter", stream_adapter)
engine.load(('./main.qml'))

sys.exit(app.exec_())