import sys
import threading
import queue
import numpy as np

from Tracker.HandTracker import HandTracker, cv2

from PyQt5.QtWidgets import (QApplication, QWidget, QDesktopWidget, QVBoxLayout, QSplitter, QLabel,
                             QTextEdit, QMainWindow, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5 import QtGui
from PyQt5 import QtCore

running = False
capture_thread = None
q = queue.Queue()
q_erosion = queue.Queue()
q_dilation = queue.Queue()
isFirstFrame = True


def grab(cam, queue, q_erosion, q_dilation, width, height, fps):
    global running
    kernel = np.ones((10, 10), np.uint8)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while(running):
        frame = {}
        frame_erosion = {}
        frame_dilation = {}
        success, img = capture.read()

        img_erosion = cv2.erode(img, kernel, iterations = 1)
        imd_dilation = cv2.dilate(img, kernel, iterations = 1)

        if success:
            frame["img"] = img
            frame_erosion["img"] = img_erosion
            frame_dilation["img"] = imd_dilation

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            queue.qsize()

        if q_erosion.qsize() < 10:
            q_erosion.put(frame_erosion)
        else:
            q_erosion.qsize()

        if q_dilation.qsize() < 10:
            q_dilation.put(frame_dilation)
        else:
            q_dilation.qsize()


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

    def getImage(self):
        return self.image



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = HandTracker(4)
        self.title = 'Finger Spelling'
        self.resize(900, 500)
        self.center()
        self.setWindowTitle(self.title)
        self.count = 0
        self.use_feed = False

        self.frame_widget = FrameWidget(self)
        self.setCentralWidget(self.frame_widget)

        self.frame_widget.start_button.clicked.connect(self.start_clicked)

        self.window_width = self.frame_widget.live_feed.width()
        self.window_height = self.frame_widget.live_feed.height()
        self.win_erosion_h = self.frame_widget.live_feed_erosion.height()
        self.win_erosion_w = self.frame_widget.live_feed_erosion.width()
        self.win_dilation_h = self.frame_widget.live_feed_dilation.height()
        self.win_dilation_w = self.frame_widget.live_feed_dilation.width()

        self.frame_widget.live_feed = OwnImageWidget(self.frame_widget.live_feed)
        self.frame_widget.live_feed_erosion = OwnImageWidget(self.frame_widget.live_feed_erosion)
        self.frame_widget.live_feed_dilation = OwnImageWidget(self.frame_widget.live_feed_dilation)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.isFirstFrame = True
        self.show()

    def getLiveFeed(self):
        liveFeed = None
        if self.frame_widget.live_feed is not None:
            liveFeed = self.frame_widget.live_feed

        return liveFeed

    def start_clicked(self):
        global running
        running = True
        capture_thread = threading.Thread(target=grab, args=(0, q, q_erosion, q_dilation, 1280, 720, 30))
        capture_thread.start()

    def closeEvent(self, event):
        global running
        running = False

    def set_isFirstFrame(self, isFirst):
        self.isFirstFrame = isFirst

    def update_frame(self):
        global running
        if not q.empty():
            frame = q.get()
            img = frame["img"]
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            toPut = None

            if running:
                if self.isFirstFrame:
                    if self.frame_widget.live_feed is not None:
                        toPut = self.tracker.initTracker(self.frame_widget.live_feed, frame)
                else:
                    print('Not first frame')
                    if self.frame_widget.live_feed is not None:
                        toPut = self.tracker.trackframe(self.frame_widget.live_feed, frame)

            if toPut is not None:
                resizedImg = cv2.resize(toPut, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
                height, width, channel = img.shape
                bytesPerLine = 3*width
                qtImage = QtGui.QImage(img, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                self.frame_widget.live_feed.setImage(qtImage)

            self.set_isFirstFrame(False)

        if not q_erosion.empty():
            frame_erosion = q_erosion.get()
            img_erosion = frame_erosion["img"]

            self.count += 1
            if self.count > 5:
                self.use_feed = True
                self.count = 0

            if running:
                if self.isFirstFrame is False:
                    if self.frame_widget.live_feed_erosion is not None:
                        if self.use_feed is True:
                            state = self.tracker.get_tracking_state()
                            if state is True:
                                p1 = self.tracker.get_corner()
                                p2 = self.tracker.get_opposite_corner()
                                p1_y = p1[1]
                                p1_y_h = p2[1] - p1[1]
                                p1_x = p1[0]
                                p1_x_w = p2[0] - p1[0]
                                img_erosion = img_erosion[p1_y:p1_y+p1_y_h, p1_x:p1_x+p1_x_w]
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            else:
                pass

            img_height_e, img_width_e, img_colors_e = img_erosion.shape

            scale_w_e = float(self.win_erosion_w) / float(img_width_e)
            scale_h_e = float(self.win_erosion_h) / float(img_height_e)
            scale_e = min([scale_w_e, scale_h_e])

            if scale_e == 0:
                scale_e = 1

            resizedImg_e = cv2.resize(img_erosion, None, fx=scale_e, fy=scale_e, interpolation=cv2.INTER_CUBIC)
            img_e = cv2.cvtColor(resizedImg_e, cv2.COLOR_BGR2RGB)
            height_e, width_e, channel_e = img_e.shape
            bytesPerLine_e = 3 * width_e
            qtImage_e = QtGui.QImage(img_e.data, width_e, height_e, bytesPerLine_e, QtGui.QImage.Format_RGB888)
            self.frame_widget.live_feed_erosion.setImage(qtImage_e)

        if not q_dilation.empty():
            frame_dilation = q_dilation.get()
            img_dilation = frame_dilation["img"]

            if running:
                if self.isFirstFrame is False:
                    if self.frame_widget.live_feed_dilation is not None:
                        if self.use_feed is True:
                            state = self.tracker.get_tracking_state()
                            if state is True:
                                p1 = self.tracker.get_corner()
                                p2 = self.tracker.get_opposite_corner()
                                p1_y = p1[1]
                                p1_y_h = p2[1] - p1[1]
                                p1_x = p1[0]
                                p1_x_w = p2[0] - p1[0]
                                img_dilation = img_dilation[p1_y:p1_y+p1_y_h, p1_x:p1_x+p1_x_w]
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            else:
                pass

            img_height_d, img_width_d, img_colors_d = img_dilation.shape

            scale_w_d = float(self.win_dilation_w) / float(img_width_d)
            scale_h_d = float(self.win_dilation_h) / float(img_height_d)
            scale_d = min([scale_w_d, scale_h_d])

            if scale_d == 0:
                scale_d = 1

            resizedImg_d = cv2.resize(img_dilation, None, fx=scale_d, fy=scale_d, interpolation=cv2.INTER_CUBIC)
            img_d = cv2.cvtColor(resizedImg_d, cv2.COLOR_BGR2RGB)
            height_d, width_d, channel_d = img_d.shape
            bytesPerLine_d = 3 * width_d
            qtImage_d = QtGui.QImage(img_d.data, width_d, height_d, bytesPerLine_d, QtGui.QImage.Format_RGB888)
            self.frame_widget.live_feed_dilation.setImage(qtImage_d)


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
        self.live_feed_erosion.setMinimumSize(200, 200)
        self.live_feed_erosion.setMaximumSize(200,200)
        self.live_feed_dilation = QWidget()
        self.live_feed_dilation.setMinimumSize(200, 200)
        self.live_feed_dilation.setMaximumSize(200,200)

        self.side_window = QSplitter(Qt.Vertical)
        self.side_window.addWidget(self.live_feed_erosion)
        self.side_window.addWidget(self.live_feed_dilation)

        self.upper_frame = QSplitter(Qt.Horizontal)
        self.upper_frame.addWidget(self.live_feed)
        self.upper_frame.addWidget(self.side_window)

        self.start_button = QPushButton('Start feed', self)
        self.train_nnetwork = QPushButton('Train neural network', self)

        self.label = QLabel("Output: ")
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumSize(850, 100)
        self.text_edit.setFont(QFont("Ariel", 14))

        self.b_splitter = QSplitter(Qt.Vertical)
        self.b_splitter.addWidget(self.label)
        self.b_splitter.addWidget(self.start_button)

        self.bb_splitter = QSplitter(Qt.Vertical)
        self.bb_splitter.addWidget(self.b_splitter)
        self.bb_splitter.addWidget(self.train_nnetwork)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.bb_splitter)
        self.splitter.addWidget(self.text_edit)

        self.layout.addWidget(self.upper_frame)
        self.layout.addWidget(self.splitter)

        self.setLayout(self.layout)

