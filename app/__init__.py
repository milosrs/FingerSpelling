from Tracker.HandTracker import HandTracker
from View.MainWindow import *


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
