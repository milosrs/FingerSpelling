import sys
import cv2
import threading
import queue

from PyQt5.QtWidgets import (QApplication, QWidget, QDesktopWidget, QVBoxLayout, QSplitter, QLabel,
                             QTextEdit, QMainWindow, QPushButton)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5 import QtGui
from PyQt5 import QtCore

running = False
capture_thread = None
q = queue.Queue()


def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while(running):
        frame = {}
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            queue.qsize()


class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Finger Spelling'
        self.resize(900, 500)
        self.center()
        self.setWindowTitle(self.title)

        self.frame_widget = FrameWidget(self)
        self.setCentralWidget(self.frame_widget)

        self.frame_widget.start_button.clicked.connect(self.start_clicked)

        self.window_width = self.frame_widget.live_feed.width()
        self.window_height = self.frame_widget.live_feed.height()
        self.frame_widget.live_feed = OwnImageWidget(self.frame_widget.live_feed)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.show()


    def start_clicked(self):
        global running
        running = True
        capture_thread.start()

    def closeEvent(self, event):
        global running
        running = False

    def update_frame(self):
        if not q.empty():
            frame = q.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.frame_widget.live_feed.setImage(image)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class FrameWidget(QWidget):
    # noinspection PyArgumentList
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout()

        self.live_feed = QWidget()
        self.live_feed_erosion = QWidget()
        self.live_feed_erosion.setMaximumSize(200,200);
        self.live_feed_dilation = QWidget()
        self.live_feed_dilation.setMaximumSize(200,200)

        self.side_window = QSplitter(Qt.Vertical)
        self.side_window.addWidget(self.live_feed_erosion)
        self.side_window.addWidget(self.live_feed_dilation)

        self.upper_frame = QSplitter(Qt.Horizontal)
        self.upper_frame.addWidget(self.live_feed)
        self.upper_frame.addWidget(self.side_window)

        self.start_button = QPushButton('Start feed', self)

        self.label = QLabel("Output: ")
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumSize(850, 100)
        self.text_edit.setFont(QFont("Ariel", 14))

        self.b_splitter = QSplitter(Qt.Vertical)
        self.b_splitter.addWidget(self.label)
        self.b_splitter.addWidget(self.start_button)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.b_splitter)
        self.splitter.addWidget(self.text_edit)

        self.layout.addWidget(self.upper_frame)
        self.layout.addWidget(self.splitter)

        self.setLayout(self.layout)

capture_thread = threading.Thread(target=grab, args = (0, q, 1280, 720, 30))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
