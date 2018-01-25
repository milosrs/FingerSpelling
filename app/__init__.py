from Tracker.HandTracker import HandTracker
import cv2


def main():
    tracker = HandTracker(2)
    tracker.printVersions()
    tracker = cv2.TrackerGOTURN_create();


if __name__ == "__main__":
    main()
