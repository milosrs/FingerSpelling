import scipy.io
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

validationImagesPath = '../hand_dataset/validation_dataset/validation_data/images/'
validationAnnotationsPath = '../hand_dataset/validation_dataset/validation_data/annotations/'

testImagesPath = '../hand_dataset/test_dataset/test_data/images/'
testAnnotationsPath = '../hand_dataset/test_dataset/test_data/annotations/'

valid_image_formats = ["jpg", "gif", "png", "bmp"]
valid_annot_formats = ["mat"]

class Relation:
    def __init__(self, image, annotation, ROI):
        self.imageFilename = image
        self.imagePath = join(testImagesPath, image)
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
            rect = patches.Rectangle((x_min, y_max),x_max-x_min, y_min-y_max,linewidth=1, edgecolor='r', facecolor='none')
            axes.add_patch(rect)
        plot.show()

    def generateXML(self):
        xmlPath = join('../hand_dataset/test_dataset/test_data/xmlAnnotations/',
                            self.imageFilename.split('.')[0])
        xmlPath = xmlPath+'.xml'
        print(xmlPath)
        height, width, channels = self.image.shape

        #Root
        root = etree.Element('annotation')

        #Folder
        folder = etree.Element('folder')
        folder.text = testImagesPath

        #File
        filename = etree.Element('filename')
        filename.text = self.imageFilename

        #Size
        sizeAnot = etree.Element('size')
        widthAnot = etree.Element('width')
        heightAnot = etree.Element('height')
        depth = etree.Element('depth')
        widthAnot.text = str(width)
        heightAnot.text = str(height)
        depth.text = str(channels)
        sizeAnot.append(widthAnot)
        sizeAnot.append(heightAnot)
        sizeAnot.append(depth)

        #Segmented
        segmented = etree.Element('segmented')
        segmented.text = str(0)

        #Root appendings
        root.append(folder)
        root.append(filename)
        root.append(sizeAnot)
        root.append(segmented)

        for coordList in self.ROIdata:
            obj = etree.Element('object')
            name = etree.Element('name')
            name.text = 'Hand'
            pose = etree.Element('pose')
            pose.text = 'Unspecified'
            truncated = etree.Element('truncated')
            truncated.text = str(0)
            difficult = etree.Element('difficult')
            difficult.text = str(0)
            bndbox = etree.Element('bndbox')
            xmin = etree.Element('xmin')
            xmax = etree.Element('xmax')
            ymin = etree.Element('ymin')
            ymax = etree.Element('ymax')
            xmax.text = str(max(coordList[0][1], coordList[1][1], coordList[2][1], coordList[3][1]))
            xmin.text = str(min(coordList[0][1], coordList[1][1], coordList[2][1], coordList[3][1]))
            ymax.text = str(max(coordList[1][0], coordList[2][0], coordList[0][0], coordList[3][0]))
            ymin.text = str(min(coordList[1][0], coordList[2][0], coordList[0][0], coordList[3][0]))
            bndbox.append(xmin)
            bndbox.append(xmax)
            bndbox.append(ymin)
            bndbox.append(ymax)
            obj.append(name)
            obj.append(pose)
            obj.append(truncated)
            obj.append(difficult)
            obj.append(bndbox)
            root.append(obj)
        tree = etree.ElementTree(root)
        with open(xmlPath, 'wb') as file:
            toWrite = etree.tostring(tree, pretty_print=True)
            file.write(toWrite)

class ImagesProcessing:
    def __init__(self):
        self.NUM_CLASSES = 1
        self.IMAGE_NUMBER = 4069
        self.ANNOTATION_NUMBER = 4069

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
        images = [img for img in listdir(testImagesPath) if isfile(join(testImagesPath, img))]
        annotations = [mat for mat in listdir(testAnnotationsPath) if isfile(join(testAnnotationsPath, mat))]
        numberOfFiles = 0

        while not images.__len__() == 0 and not annotations.__len__() == 0:
            imageToBeUsed = images.pop()
            annotationToBeUsed = join(testAnnotationsPath, annotations.pop())

            if self.checkFormat(imageToBeUsed, valid_image_formats) and self.checkFormat(annotationToBeUsed, valid_annot_formats):
                ROI = self.createRoiFromAnnotation(annotationToBeUsed)
                testRelation = Relation(imageToBeUsed, annotationToBeUsed, ROI)
                #testRelation.plotWithBBox()
                testRelation.generateXML()
                numberOfFiles+=1

        return numberOfFiles

    def getRelationCollection(self):
        return self.relationCollection

imgproc = ImagesProcessing()
numberOfFiles = imgproc.createRelations()
print("Creating XML Files over! Files created: {0}".format(numberOfFiles))