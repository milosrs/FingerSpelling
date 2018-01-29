import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as patches
from PIL import Image

from os import listdir
from os.path import isfile, join
from lxml import etree

#Acces to ROI like this
# roiShape = self.ROI.shape()
# for i in range(roiShape[0])
#   for j in range(roiShape[1])
#       ...Code here

trainingImagesPath = '../hand_dataset/training_dataset/training_data/images/'
trainingAnnotationsPath = '../hand_dataset/training_dataset/training_data/annotations/'
valid_image_formats = ["jpg", "gif", "png", "bmp"]
valid_annot_formats = ["mat"]

class Relation:
    def __init__(self, image, annotation, ROI):
        self.imageFilename = image
        self.imagePath = join(trainingImagesPath, image)
        self.annotation = annotation
        self.ROIdata = ROI
        self.ROI = []
        self.image = np.array(Image.open(self.imagePath), dtype=np.uint8)

    def printProperties(self):
        print("Properties: {0} \n Coords: {1}".format(self.imagePath, self.ROIdata))

    def plotWithBBox(self):
        fig, axes = plot.subplots(1)
        axes.imshow(self.image)
        for coordList in self.ROIdata:
            x_max = max(coordList[0][1],coordList[1][1],coordList[2][1],coordList[3][1])
            x_min = min(coordList[0][1],coordList[1][1],coordList[2][1],coordList[3][1])
            y_max = max(coordList[1][0], coordList[2][0],coordList[0][0], coordList[3][0])
            y_min =  min(coordList[1][0], coordList[2][0],coordList[0][0], coordList[3][0])
            print("X_MIN: {0}, X_MAX: {1}, Y_MIN: {2}, Y_MAX: {3}".format(x_min, x_max, y_min, y_max))
            rect = patches.Rectangle((x_min, y_max),x_max-x_min, y_min-y_max,linewidth=1, edgecolor='r', facecolor='none')
            axes.add_patch(rect)
        plot.show()

    def generateXML(self):
        root = etree.Element('annotation')
        folder = etree.Element('folder')
        folder.text = trainingImagesPath
        filename = etree.Element('filename')
        filename.text = self.imagePath
        size = self.image.shape()
        print(size)
        #for coordList in self.ROIdata:
           # x_max = max(coordList[0][1], coordList[1][1], coordList[2][1], coordList[3][1])
           # x_min = min(coordList[0][1], coordList[1][1], coordList[2][1], coordList[3][1])
          #  y_max = max(coordList[1][0], coordList[2][0], coordList[0][0], coordList[3][0])
           # y_min = min(coordList[1][0], coordList[2][0], coordList[0][0], coordList[3][0])
          #  width = x_max-x_min
           # height = (y_max - y_min)


class ImagesProcessing:
    def __init__(self):
        self.NUM_CLASSES = 1
        self.IMAGE_NUMBER = 4069
        self.ANNOTATION_NUMBER = 4069
        self.relationCollection = []

    def checkFormat(self, fileName, formats):
        passed = False
        fileName = fileName[3:]
        splitres = fileName.split('.')
        for splits in splitres:
            if splits in formats:
                passed = True
                break

        return passed

    def testEquality(self, collection):
        if collection.__len__() == self.IMAGE_NUMBER:
            return True
        else:
            return False

    def createRoiFromAnnotation(self, annotation):
        mat = scipy.io.loadmat(annotation)
        ret = mat['boxes']
        shapeR,shapeC = np.shape(ret)
        ROI = []
        for i in range(0,shapeR):
            for j in range (0, shapeC):
                a = ret[i, j][0][0][0][0]
                b = ret[i, j][0][0][1][0]
                c = ret[i, j][0][0][2][0]
                d = ret[i, j][0][0][3][0]
                toAppend = [a,b,c,d]
                ROI.append(toAppend)

        return ROI

    def createRelations(self):
        images = [img for img in listdir(trainingImagesPath) if isfile(join(trainingImagesPath, img))]
        annotations = [mat for mat in listdir(trainingAnnotationsPath) if isfile(join(trainingAnnotationsPath, mat))]

        while not images.__len__() == 0 and not annotations.__len__() == 0:
            imageToBeUsed = images.pop()
            annotationToBeUsed = join(trainingAnnotationsPath, annotations.pop())

            if self.checkFormat(imageToBeUsed, valid_image_formats) and self.checkFormat(annotationToBeUsed, valid_annot_formats):
                ROI = self.createRoiFromAnnotation(annotationToBeUsed)
                testRelation = Relation(imageToBeUsed, annotationToBeUsed, ROI)
                testRelation.printProperties()
                #testRelation.plotWithBBox()
                testRelation.generateXML()
                self.relationCollection.insert(self.relationCollection.__len__(), testRelation)

    def getRelationCollection(self):
        return self.relationCollection

imgproc = ImagesProcessing()
imgproc.createRelations()
print("BROJ RELACIJA:")
print(imgproc.getRelationCollection().__len__())