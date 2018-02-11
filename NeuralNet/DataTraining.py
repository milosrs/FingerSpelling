#TREBA DA SE ODRADI TRANSFER ZNANJA PREKO INCEPTIONA!
#TREBA DA ISTRENIRAMO NEURONSKU KORISTECI MNIST, CIFAR10, RUKE....
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/08_Transfer_Learning.ipynb


#Za klasifikaciju sekvence koristiti Stacked LSTM mreze! (Long Short Term Memmory Network)
#Postoji ceo clanak o tome na Kerasu

import math
import matplotlib.pyplot as plot
import tensorflow
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import pylab as pl
import matplotlib.cm as cm
import numpy.ma as ma
from PIL import Image
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle
from os import listdir
from enum import Enum
from os.path import isfile, join
from tensorflow.examples.tutorials.mnist import input_data

class TrainingBatch(Enum):
    CIFAR='CIFAR'
    MNIST='MNIST'
    HANDS='HANDS'
    CIFAR10='CIFRA10'


activeTrainingBatch = TrainingBatch.HANDS


#Size of MNIST image
img_size_MNIST = 28

#Size of CIFAR-10 image
img_size_CIFAR = 32

#Active images from dataset size
img_size = img_size_CIFAR

#They are stored in 1D array of this length
img_size_flat = img_size*img_size

#Shape of image
img_shape = (img_size, img_size)

#3D image shape (Z dimension = 1)
img_shape_full = (img_size, img_size, 1)

#Color channels
if img_size == img_size_CIFAR:
    num_channels = 3
elif img_size == img_size_MNIST:
    num_channels  = 1

#Number of classes (Numbers [0-9])
num_classes = 10

#Serialized net model
modelPath = 'netModel.keras'

def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Net():
    def __init__(self, weights_path=None, batch_size=None):
        self.model = self.create_model(weights_path, batch_size)
        self.optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.batch_size = batch_size

    #Creates a new deep model.
    def create_model(self, weights_path, batch_size):
        model = Sequential()

        #Slike n*224x224x3 (3=kanali, n=broj slika)
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))

        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(24, activation='softmax'))          #24 klase

        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)

        return model

    #Saves and deletes an existing model.
    def save_model(self, path):
        self.model.save(path)
        del self.model
        self.model = None

    def start_training(self, images, labels):
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x=images, y=labels, epochs=70, validation_split=0.1, shuffle=True, verbose=1)
        return history

    def load_model(self, modelPath):
        self.model = load_model(modelPath)

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


class Ploter():

    def plot_images(self, images, cls_true, cls_pred=None):
        assert len(images) == len(cls_true) == 9

        # Create figure with 3x3 subplots
        fig, axes = plot.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot image
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes
            if cls_pred is None:
                 xlabel = 'True: {0}'.format(cls_true[i])
            else:
                 xlabel = 'True: {0}, Pred: {1}'.format(cls_true[i], cls_pred[i])
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        plot.show()

    def plot_errors(self, cls_pred):
        incorrect = (cls_pred != data.test.classNo)
        incorrect_imgs = data.test.images[incorrect]
        cls_pred = cls_pred[incorrect]
        cls_true = data.test.classNo[incorrect]

        self.plot_images(images=incorrect_imgs[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

    def plot_weights(self, weights, input_channel=0):
        #Max and min weigth for the correct color gradient
        w_min = np.min(weights)
        w_max = np.max(weights)

        #number of filters used in conv layer
        num_filters = weights.shape[3]
        #number of grids to draw
        num_grids = math.ceil(math.sqrt(num_filters))

        fig,axes = plot.subplots(num_grids, num_grids)

        for i, ax in enumerate(axes.flat):
            if i<num_filters:
                img = weights[:,:,input_channel,i]
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('Visualised weights')

        plot.show()

    def plot_conv_output(self, values):
        #Filters used in conv layer
        num_filters = values.shape[3]
        num_grids = math.ceil(math.sqrt(num_filters))
        fig, axes = plot.subplots(num_grids, num_grids)

        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                img = values[0, :, :, i]
                ax.imshow(img, interpolation='nearest', cmap='binary')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('Visualised conv output')

        plot.show()

    def plot_image(self, image):
        plot.imshow(image.reshape(img_shape),
                   interpolation='nearest',
                   cmap='binary')
        plot.show()

    def plot_learning_process(self, history):
        #accuraccy
        keys = history.history.keys()
        print(keys)
        for key in keys:
            plot.plot(history.history[key])
            plot.title('model '+key)
            plot.ylabel(key)
            plot.xlabel('epoch')
            plot.legend(['train', 'test'], loc='upper left')
            plot.show()

        plot.show()

class ImageConverter():

    def __init__(self):
        self.labelmatrix = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'e': 0, 'i': 0, 'j': 0, 'k': 0, 'l': 0,
            'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 0, 't': 0, 'u': 0, 'v': 0, 'w': 0, 'x': 0, 'y': 0, 'z': 0}
        self.labelOnehot = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8,
                            'k': 9, 'l': 10,
                            'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't':18, 'u': 19, 'v': 20, 'w': 21,
                            'x': 22, 'y': 23}
        self.trainingdata = []
        self.traininglabels = []
        self.path = '../../dataset/'
        self.trainingName = 'trainingBatches/trainingBatch'
        self.labelName = 'trainingBatches/labelBatch'
        self.trainingDataNumber = 0
        self.readFilenames = []
        self.batchNumber = 0

    #Upozorenje: 2GB Rama ce zauzeti slike.
    def createTrainingData(self):
        if os.path.exists(self.trainingName):
            return

        originalPath = '../../dataset5/'
        paths = listdir(originalPath)
        label = ''
        converter = ImageConverter()
        max_pic_per_folder = 70    # Imamo 5 foldera sa 24 subfoldera i svaki subfolder 3000 slika.... 65000 slika
        read_pictures_per_folder = 0

        while len(self.readFilenames) <= 65001:
            for path in paths:
                pathToGo = join(originalPath, path)
                pathsinPaths = listdir(pathToGo)
                for letter in pathsinPaths:
                    pathInPath = pathToGo + "/" + letter
                    label = letter
                    printed = False
                    for imagePath in glob.glob(pathInPath + '/*.png'):
                        if imagePath not in self.readFilenames and read_pictures_per_folder != max_pic_per_folder:
                            if printed is False:
                                print(imagePath)
                                printed = True

                            image = cv2.imread(imagePath)
                            image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LANCZOS4)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            npImg = np.asarray(image, dtype=np.uint8)
                            self.trainingdata.append(npImg)
                            self.traininglabels.append(self.labelOnehot[label])
                            read_pictures_per_folder += 1
                            self.readFilenames.append(imagePath)
                            #self.cannyImages(list_of_images, letter)
                            #del list_of_images
                        elif read_pictures_per_folder == max_pic_per_folder:
                            print(imagePath+" : "+str(read_pictures_per_folder))
                            read_pictures_per_folder = 0
                            printed = False
                            break

                    print("End of traversal")
            self.saveShuffledData()
            self.batchNumber += 1
            self.traininglabels.clear()
            self.trainingdata.clear()
            print("Broj batcha:"+str(self.batchNumber))

    def cannyImages(self, imageArray, label):
        for img in imageArray:
            resized = cv2.resize(img, (200,200))
            smoothimg = cv2.GaussianBlur(resized, (5,5), 0)             #Koristimo gaussian blur za otklanjanje gaus noise
            edged = cv2.Canny(smoothimg, 70, 150)
            foldername = join(self.path, label+"/")
            shortname = "color_" + str(self.labelmatrix[label]) + ".png"
            self.labelmatrix[label] = self.labelmatrix[label] + 1
            filename = join(foldername, shortname)

            if not os.path.exists(foldername):
                os.makedirs(foldername)

            cv2.imwrite(filename, edged)

    def saveShuffledData(self):
        imgBatchPath = self.trainingName+str(self.batchNumber)+".npy"
        labelBatchPath = self.labelName+str(self.batchNumber)+"npy"
        print("Saving data...")

        if os.path.exists(imgBatchPath):
            self.trainingdata = np.load(imgBatchPath)
            self.traininglabels = np.load(labelBatchPath)
            return

        self.trainingdata, self.traininglabels = shuffle(self.trainingdata, self.traininglabels, random_state=0)
        np.save(labelBatchPath, self.traininglabels)
        np.save(imgBatchPath, self.trainingdata)

    def loadBatch(self, path):
        return np.load(path)

    def getTrainingData(self):
        return self.trainingdata

    def setTrainingData(self, td):
        self.trainingdata = td

    def getTrainingLabels(self):
        return self.traininglabels

    def destroyLists(self):
        del self.trainingdata

def openLabelFile(batchNumber, files):
    for file in files:
        if "label" in file:
            if str(batchNumber) in file:
                return file

ploter = Ploter()

if activeTrainingBatch == TrainingBatch.MNIST:
    model = Net()
    data = input_data.read_data_sets('MNIST', one_hot=True)
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(len(data.test.labels)))
    print("- Validation-set:\t{}".format(len(data.validation.labels)))
    print('Tensorflow version: '+tensorflow.__version__)
    print('Keras: '+tensorflow.keras.__version__)
    data.test.classNo = np.argmax(data.test.labels, axis=1)

    #Creating model for training. Loss function = https://en.wikipedia.org/wiki/Cross_entropy
    #Read documentation for fit.

    if isfile(modelPath):
        model.set_model(load_model(modelPath))
    else:
        history = model.start_training(data.test.images, data.test.labels)
        model.save_model('netModel.keras')
        ploter.plot_learning_process(history)
        result = model.get_model().evaluate(x=data.test.images, y=data.test.labels)
        for name, value in zip(model.get_model().metrics_names, result):
            print(name, value)
            print("{0}: {1:.2%}".format(model.get_model().metrics_names[1], result[1]))

    testimages = data.test.images[0:9]
    cls_true = data.test.classNo[0:9]
    predictions = model.get_model().predict(x=testimages)
    cls_pred = np.argmax(predictions, axis=1)
    cls_true = data.test.classNo[0:9]
    ploter.plot_images(testimages, cls_pred=cls_pred, cls_true=cls_true)

elif activeTrainingBatch == TrainingBatch.CIFAR:
    model = Net()
    dictionary = unpickle('cifar-10-batches-py/data_batch_1')
    print(dictionary.keys())
    cifimages = dictionary[b'data']
    ciflabels = dictionary[b'labels']
    ciflabelnames = dictionary[b'batch_label']

    testDict = unpickle('cifar-10-batches-py/test_batch')
    testimg = testDict[b'data']
    testlabel = testDict[b'labels']
    testlabelnames = testDict[b'batch_label']

    if isfile(modelPath):
        model.set_model(load_model(modelPath))
    else:
        history = model.start_training(cifimages, ciflabels)
        model.save_model('netModel.keras')
        ploter.plot_learning_process(history)
        result = model.get_model().evaluate(x=testimg, y=testlabel)
        for name, value in zip(model.get_model().metrics_names, result):
            print(name, value)
            print("{0}: {1:.2%}".format(model.get_model().metrics_names[1], result[1]))

elif activeTrainingBatch == TrainingBatch.HANDS:

    batchpath = "trainingBatches/"
    modelPath = "letterDetector.h5"
    files = os.listdir(batchpath)
    fileNo = math.floor(len(files)/2)
    imageconverter = ImageConverter()
    nextBatch = 0
    batchesTrainedOnPath = "batchesTrainedOn.npy"

    if(fileNo == 0):
       #imageconverter.writeConvertedImages()              Prebacivanje slike u samo ivice (Koristi se za random forest)
        imageconverter.createTrainingData()
        imageconverter.saveShuffledData()
    else:
        model = Net()
        if modelPath in listdir():
            model.load_model(modelPath)
            nextBatch = np.load(batchesTrainedOnPath)[0]

        start = "Batch"
        end = ".npy"
        for file in files:
            if "training" in file:
                startStr = file.find(start) + len(start)
                endStr = file.find(end, startStr)
                batchNumber = file[startStr:endStr]

                if str(nextBatch) == batchNumber:
                    labelsFile = openLabelFile(batchNumber, files)
                else:
                    labelsFile = openLabelFile(str(nextBatch), files)

                training_data = imageconverter.loadBatch(batchpath + file)
                training_labels = imageconverter.loadBatch(batchpath + labelsFile)
                history = model.start_training(images=training_data, labels=training_labels)
                model.save_model(modelPath)
                model.load_model(modelPath)
                files.remove(file)
                files.remove(labelsFile)
                nextBatch += 1
                np.save(batchesTrainedOnPath, np.asarray(nextBatch))



