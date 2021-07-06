import sys
from PySide6.QtCore import Property, QObject, QUrl, Slot

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


class WebcamStream():
    def __init__(self):
        pass

class GUIBackend(QObject):

    def __init__(self):
        QObject.__init__(self)
        self._status = "Initialized"
        

    @Property(str)
    def status(self):
        return self._status
    

    @Slot()
    def start_webcam_feed(self):
        print("here")


app = QGuiApplication(sys.argv)

engine = QQmlApplicationEngine()
backend = GUIBackend()
engine.rootContext().setContextProperty("GUIBackend", backend)
engine.load(QUrl.fromLocalFile('./main.qml'))

sys.exit(app.exec())