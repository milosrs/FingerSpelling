import os
import tensorflow as tf
import scipy.io

from os import listdir, walk
from os.path import isfile, join
from tensorflow.contrib.factorization.examples.mnist import NUM_CLASSES

NUM_CLASSES = 1
IMAGE_NUMBER = 4069
ANNOTATION_NUMBER = 4069

def checkFormat(fileName, formats):
    passed = False
    if fileName.split('.')[1] in formats:
        passed = True
    
    return passed

def testEquality(collection):
    if collection.__len__() == IMAGE_NUMBER:
        return True
    else:
        return False
    
def createRelations():
    
    #Used for modeling image/annotation relations
    class Relation:
        
        def __init__(self, image, annotation):
            self.image = image
            self.annotation = annotation
            
        def printProperties(self):
            print("Properties: "+self.image+", "+self.annotation)
            
        def doFilenamesMatch(self):
            return self.image.split('.')[0].__eq__(self.annotation.split('.')[0])
    
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
        
        if checkFormat(imageToBeUsed, valid_image_formats) and checkFormat(annotationToBeUsed, valid_annot_formats):
            
            testRelation = Relation(imageToBeUsed, annotationToBeUsed)
            relationCollection.insert(relationCollection.__len__(), testRelation)
            
    return relationCollection
    
print(testEquality(createRelations()))