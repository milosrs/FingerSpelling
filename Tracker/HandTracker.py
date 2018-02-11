import cv2
import numpy as np

from enum import Enum
from darkflow.net.build import TFNet
from keras.models import load_model

class Signal(Enum):
    YOLO='YOLO',
    TRACK='TRACK',

class Mode(Enum):
    NO_DETECTION="N"
    DETECTION="Y"

modelPath = "../NeuralNet/letterDetector.h5"

class HandTracker():

    def __init__(self, trackType):
        self.trackerTypes = ['Boosting', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.selectedTracker = self.trackerTypes[trackType]
        (major,minor,subminor) = (cv2.__version__).split('.')
        self.major = major
        self.minor = minor
        self.subminor = subminor
        self.top_left= None
        self.bottom_right = None
        self.tracking_state = None
        self.colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
        self.signal = Signal.YOLO
        self.yolo = None
        self.handClassifier = None

    def initYOLO(self):
        options = {
            'model': 'cfg\\tiny-yolo-voc-1c.cfg',
            'load': 27125,
            'threshold': 0.1
        }

        self.yolo = TFNet(options)

    def initHandClassifier(self):
        self.handClassifier = load_model(modelPath)

    def printVersions(self):
        print('OpenCV Version: {}.{}.{}'.format(self.major, self.minor, self.subminor))

    def detect(self, firstFrame):
        frame = None
        tl = None
        br = None
        results = self.yolo.return_predict(firstFrame)

        for color, result in zip(self.colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)

            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        if frame is not None:
            numpyImage = np.asarray(frame, dtype=np.uint8)

        if tl is not None and br is not None:
            self.tracker.init(numpyImage, [tl[0], tl[1], br[0], br[1]])
            self.signal = Signal.TRACK

        return numpyImage

    def selectBB(self, firstFrame):
        try:
            bbox = cv2.selectROI(firstFrame, False)
            cv2.destroyAllWindows()
            success = self.tracker.init(firstFrame, bbox)

            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(firstFrame, p1, p2, (0, 255, 0), 2, 1)
            else:
                cv2.putText(firstFrame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            numpyImage = np.asarray(firstFrame, dtype=np.uint8)
            self.signal = Signal.TRACK
            return numpyImage
        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)



    def initTracker(self, firstFrame, mode=Mode.NO_DETECTION):
        if self.selectedTracker == self.trackerTypes[0]:
            self.tracker = cv2.TrackerBoosting_create()
        elif self.selectedTracker == self.trackerTypes[1]:
            self.tracker = cv2.TrackerMIL_create()
        elif self.selectedTracker == self.trackerTypes[2]:
            self.tracker = cv2.TrackerKCF_create()
        elif self.selectedTracker == self.trackerTypes[3]:
            self.tracker = cv2.TrackerTLD_create()
        elif self.selectedTracker == self.trackerTypes[4]:
            self.tracker = cv2.TrackerMedianFlow_create()
        elif self.selectedTracker == self.trackerTypes[5]:
            self.tracker = cv2.TrackerGOTURN_create()

        numpyImage = None

        if mode==Mode.DETECTION:
            numpyImage = self.detect(firstFrame)
        elif mode==Mode.NO_DETECTION:
            numpyImage = self.selectBB(firstFrame)


        return numpyImage

    #def classifyHand(self, frame): TODO


    def trackframe(self, source):
        signal = Signal.TRACK

        numpyImage = np.asarray(source, dtype=np.uint8)
        success, bbox = self.tracker.update(numpyImage)
        self.tracking_state = success
        bbox = list(bbox)

        if success:
            try:
                img = source
                height, width, colors = img.shape

                if bbox[0] < 0.0:
                    signal = Signal.YOLO
                if bbox[1] < 0.0:
                    signal = Signal.YOLO
                if bbox[0]+bbox[2] > width:
                    signal = Signal.YOLO
                if bbox[1]+bbox[3] > height:
                    signal = Signal.YOLO

            except Exception as e:
                print("Thread error")
                print(type(e))
                print(e.args)
                print(e)

            bbox = tuple(bbox)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            self.top_left = p1
            self.bottom_right = p2
            cv2.rectangle(numpyImage, p1, p2, (0,255,0), 2, 1)
        else:
            #signal = signal.YOLO
            cv2.putText(numpyImage, "Detecting hands...", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        retVal = numpyImage

        return retVal

    def get_tracking_state(self):
        return self.tracking_state

    def get_corner(self):
        return self.top_left

    def get_opposite_corner(self):
        return self.bottom_right

    def get_tracker_status(self):
        return self.trackerSignal