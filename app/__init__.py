from Tracker.HandTracker import HandTracker
from View.MainWindow import MainWindow, QApplication, sys, running,q


def main():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
