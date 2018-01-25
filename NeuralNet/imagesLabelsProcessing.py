from os import listdir
from os.path import isfile, join
from tensorflow.contrib.factorization.examples.mnist import NUM_CLASSES

#Used for modeling image/annotation relations
class Relation:

    def __init__(self, image, annotation):
        self.image = image
        self.annotation = annotation

    def printProperties(self):
        print("Properties: "+self.image+", "+self.annotation)

    def doFilenamesMatch(self):
        return self.image.split('.')[0].__eq__(self.annotation.split('.')[0])

class ImagesProcessing():

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