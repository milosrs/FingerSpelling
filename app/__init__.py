from View.MainWindow import MainWindow, QApplication, sys

def main():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
