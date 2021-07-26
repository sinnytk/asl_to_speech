import sys
import cv2
import os
import mediapipe as mp
from statistics import mode


from PySide2.QtCore import Property, QObject, QTimer, Signal, Slot

from PySide2.QtGui import QGuiApplication, QImage
from PySide2.QtMultimedia import QAbstractVideoSurface, QVideoFrame, QVideoSurfaceFormat
from PySide2.QtQml import QQmlApplicationEngine, QQmlDebuggingEnabler 


from model import load_model, make_inference

model = load_model('../model/custom_model_0.pt')

class AbstractStreamAdapter(QObject):
    signal_video_surface_changed = Signal(QAbstractVideoSurface)
    annotationChanged = Signal(str)

    def __init__(self):
        QObject.__init__(self)
        self._video_surface =  None
        self._surface_format = None
        self._timer = QTimer()
        self._timer.timeout.connect(self.get_frame)

        self._opencv_feed = None
        self.counter = 0

        self.sample_inferences = []
        self.hand_frame_counter = 0
        self._annotation = "No hand"

    def cv_image_to_q(self, cv_image):
        height, width, _ = cv_image.shape
        bytes_per_line = 4 * width
        q_img = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB32)
        return q_img

    def start_surface(self):
        frame, _ = self._opencv_feed.get_frame()
        q_img = self.cv_image_to_q(frame)
        self._surface_format = QVideoSurfaceFormat(q_img.size(), QVideoFrame.pixelFormatFromImageFormat(q_img.format()))
        if not self._video_surface.isFormatSupported(self._surface_format):
            self._surface_format = self._video_surface.nearestFormat(self._surface_format)
        return self._video_surface.start(self._surface_format)

    @Slot()
    def get_frame(self):
        frame, bounded_hand = self._opencv_feed.get_frame()
        if bounded_hand is not None:
            if self.annotation  == "No hand":
                self.annotation = ''
            self.hand_frame_counter+=1
            
            if self.hand_frame_counter % 3 == 0:
                label = make_inference(model, bounded_hand)
                print(label)
                self.sample_inferences.append(label)
            if self.hand_frame_counter % 30 == 0:
                majority_annotation = mode(self.sample_inferences)
                majority_count = self.sample_inferences.count(majority_annotation)
                if majority_count > int(0.7 * len(self.sample_inferences)):
                    print(f"Majority count of {majority_annotation} is greater than 70%")
                    self.annotation += majority_annotation
                else:
                    print(f"Majority count of {majority_annotation} is less than 70%")
                self.sample_inferences = []
        else:
            self.hand_frame_counter = 0
            self.annotation = "No hand"
            self.sample_inferences = []
        q_img = self.cv_image_to_q(frame)
        a = QVideoFrame(q_img)
        self._video_surface.present(a)

    @Property(QAbstractVideoSurface, notify=signal_video_surface_changed)
    def videoSurface(self):
        return self._video_surface

    @Property(str, notify=annotationChanged)
    def annotation(self):
        return self._annotation
        
    @annotation.setter
    def annotation(self, newAnnotation):
        if self._annotation == newAnnotation:
            return
        self._annotation = newAnnotation
        self.annotationChanged.emit(self._annotation)

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
    def __init__(self):
        self.cap = None
        self.open_camera()

    def get_frame(self):
        _, frame = self.cap.read()
        bounded_hand = self.bound_hand(frame)
                    
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        return frame, bounded_hand

    def bound_hand(self, frame):
        h, w, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(framergb)

        hand_landmarks = result.multi_hand_landmarks
        handedness = result.multi_handedness

        hand_frame = None
        if hand_landmarks and handedness[0].classification[0].label == 'Left':
            wrist_landmark = hand_landmarks[0].landmark
            wrist_origin = (int(wrist_landmark[0].x * w), int(wrist_landmark[0].y * h))

            rect_start_point = (wrist_origin[0]-100, wrist_origin[1]-200)
            rect_end_point = (wrist_origin[0]+120, wrist_origin[1]+50)
            if rect_start_point[0] > 0 and rect_start_point [1] > 0:
                cv2.rectangle(frame, rect_start_point, rect_end_point, (0, 255, 0), 2)
                hand_frame = frame[rect_start_point[1]:rect_end_point[1], rect_start_point[0]:rect_end_point[0]].copy()
        return hand_frame

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