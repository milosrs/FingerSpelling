import cv2

class HandTracker():
   
    def __init__(self, trackType):
        self.trackerTypes = ['Boosting', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.selectedTracker = self.trackerTypes[trackType]
        (major,minor,subminor) = (cv2.__version__).split('.')
        self.major = major
        self.minor = minor
        self.subminor = subminor
        self.initTracker()
        
    def printVersions(self):
        print('OpenCV Version: {}.{}.{}'.format(self.major, self.minor, self.subminor))

    def initTracker(self):
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