from os import listdir
from os.path import isfile, join
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plot

#Acces to ROI like this
# roiShape = self.ROI.shape()
# for i in range(roiShape[0])
#   for j in range(roiShape[1])
#       ...Code here

def bgrtorgb(image):
    return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

def plot_image(image, figsize=(8,8), recolour=False):
    if recolour:
        image = bgrtorgb(image)
    plot.figure(figsize=figsize)
    if image.shape[-1] == 3:
        plot.imshow(image)
    elif image.shape[-1] == 1 or len(image.shape)==2:
        plot.imshow(image, cmap='gray')
    else:
        raise Exception('Image has an invalid shape')

class Relation:
    def __init__(self, image, annotation, ROI):
        self.image = image
        self.annotation = annotation
        self.ROIdata = ROI
        self.ROI = []

    def printProperties(self):
        print("Properties: "+self.image+", "+self.ROI)

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
        valid_image_formats = ["jpg","gif","png","bmp"]
        valid_annot_formats = ["mat"]
        trainingImagesPath = '../hand_dataset/training_dataset/training_data/images/'
        trainingAnnotationsPath = '../hand_dataset/training_dataset/training_data/annotations/'

        images = [img for img in listdir(trainingImagesPath) if isfile(join(trainingImagesPath, img))]
        annotations = [mat for mat in listdir(trainingAnnotationsPath) if isfile(join(trainingAnnotationsPath, mat))]

        while not images.__len__() == 0 and not annotations.__len__() == 0:
            imageToBeUsed = join(trainingImagesPath, images.pop())
            annotationToBeUsed = join(trainingAnnotationsPath, annotations.pop())

            if self.checkFormat(imageToBeUsed, valid_image_formats) and self.checkFormat(annotationToBeUsed, valid_annot_formats):
                ROI = self.createRoiFromAnnotation(annotationToBeUsed)
                testRelation = Relation(imageToBeUsed, annotationToBeUsed, ROI)
                self.relationCollection.insert(self.relationCollection.__len__(), testRelation)

    def getRelationCollection(self):
        return self.relationCollection

imgproc = ImagesProcessing()
imgproc.createRelations()
print("BROJ RELACIJA:")
print(imgproc.getRelationCollection().__len__())