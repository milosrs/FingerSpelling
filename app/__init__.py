from View.MainWindow import MainWindow, QApplication, sys
from NeuralNet.imagesLabelsProcessing import ImagesProcessing

def main():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    imageProcessor = ImagesProcessing()
    imageProcessor.createRelations()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
