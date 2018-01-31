import cv2
import numpy as np


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

    def printVersions(self):
        print('OpenCV Version: {}.{}.{}'.format(self.major, self.minor, self.subminor))

    def initTracker(self, camWindow, firstFrame):
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

        numpyImage = np.asarray(firstFrame["img"], dtype=np.uint8)
        self.bbox = cv2.selectROI(numpyImage)
        self.tracker.init(numpyImage, self.bbox)
        cv2.destroyAllWindows()
        return numpyImage

    def trackframe(self, window, source):
        numpyImage = np.asarray(source["img"], dtype=np.uint8)
        success, bbox = self.tracker.update(numpyImage)
        self.tracking_state = success
        bbox = list(bbox)

        if success:
            try:
                img = source["img"]
                height, width, colors = img.shape

                if bbox[0] < 0.0:
                    bbox[0] = 5.0
                if bbox[1] < 0.0:
                    bbox[1] = 5.0
                if bbox[0]+bbox[2] > width:
                    bbox[2] = width - bbox[0]
                if bbox[1]+bbox[3] > height:
                    bbox[3] = height - bbox[1]

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
            cv2.putText(numpyImage, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        return numpyImage

    def get_tracking_state(self):
        return self.tracking_state

    def get_corner(self):
        return self.top_left

    def get_opposite_corner(self):
        return self.bottom_right
