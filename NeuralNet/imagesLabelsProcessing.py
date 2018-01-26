from os import listdir
from os.path import isfile, join
import cv2
from tensorflow.contrib.factorization.examples.mnist import NUM_CLASSES
import keras
from keras import backend
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plot


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
        raise Exception('Image has invalid shape')

class Relation:
    def __init__(self, image, annotation):
        self.image = image
        self.annotation = annotation

    def printProperties(self):
        print("Properties: "+self.image+", "+self.annotation)

    def doFilenamesMatch(self):
        return self.image.split('.')[0].__eq__(self.annotation.split('.')[0])

class ImagesProcessing:
    def __init__(self):
        self.NUM_CLASSES = 1
        self.IMAGE_NUMBER = 4069
        self.ANNOTATION_NUMBER = 4069

    def checkFormat(self, fileName, formats):
        passed = False
        if fileName.split('.')[1] in formats:
            passed = True

        return passed

    def testEquality(self, collection):
        if collection.__len__() == self.IMAGE_NUMBER:
            return True
        else:
            return False

    def createRelations(self):
        valid_image_formats = ["jpg","gif","png","bmp"]
        valid_annot_formats = ["mat"]
        trainingImagesPath = '../hand_dataset/training_dataset/training_data/images/'
        trainingAnnotationsPath = '../hand_dataset/training_dataset/training_data/annotations/'

        images = [img for img in listdir(trainingImagesPath) if isfile(join(trainingImagesPath, img))]
        annotations = [mat for mat in listdir(trainingAnnotationsPath) if isfile(join(trainingAnnotationsPath, mat))]

        print(images)
        print(annotations)

        relationCollection = []

        while not images.__len__() == 0 and not annotations.__len__() == 0:
            imageToBeUsed = images.pop()
            annotationToBeUsed = annotations.pop()

            if self.checkFormat(imageToBeUsed, valid_image_formats) and self.checkFormat(annotationToBeUsed, valid_annot_formats):

                testRelation = Relation(imageToBeUsed, annotationToBeUsed)
                relationCollection.insert(relationCollection.__len__(), testRelation)

        return relationCollection
