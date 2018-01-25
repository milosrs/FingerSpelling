from Tracker.HandTracker import HandTracker
from View.MainWindow import *


def main():
    tracker = HandTracker(2)
    tracker.printVersions()
    capture_thread = threading.Thread(target=grab, args = (0, q, 1280, 720, 30))
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
