#TREBA DA SE ODRADI TRANSFER ZNANJA PREKO INCEPTIONA!
#TREBA DA ISTRENIRAMO NEURONSKU KORISTECI MNIST, CIFAR10, RUKE....
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/08_Transfer_Learning.ipynb


import matplotlib.pyplot as plot
import tensorflow
import numpy as np
import math
import keras.layers.advanced_activations as activations
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import cifar10
import inception
import prettytensor as pt



from os import listdir
from enum import Enum
from os.path import isfile, join
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from cifar10 import num_classes

class TrainingBatch(Enum):
    CIFAR='CIFAR'
    MNIST='MNIST'
    HANDS='HANDS'
    CIFRA10='CIFRA10'


activeTrainingBatch = TrainingBatch.CIFRA10

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

class CNNModel():
    def __init__(self):
        self.model = self.create_model()
        self.optimizer = Adam(lr=1e-3)

    #Creates a new model. Invoked in constructor
    def create_model(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(img_size_flat,)))
        model.add(Reshape(img_shape_full))

        # First CONV with ReLu activation and MaxPooling
        model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='conv1'))
        model.add(MaxPooling2D(pool_size=3, strides=2))

        # Second CONV with LeakyReLu activation and MaxPooling
        model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same', activation='relu', name='conv2'))
        model.add(MaxPooling2D(pool_size=3, strides=2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))

        # Softmax used for classification. Theory behind it: https://en.wikipedia.org/wiki/Softmax_function
        model.add(Dense(num_classes, activation='softmax'))
        return model

    #Saves and deletes an existing model.
    def save_model(self, path):
        self.model.save(path)
        del self.model
        self.model = None

    def start_training(self, images, labels):
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x=images, y=labels, epochs=3, batch_size=128)
        return history

    def plot_learning_process(self):
        self.model.built

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

ploter = Ploter()
model = CNNModel()

if activeTrainingBatch == TrainingBatch.MNIST:
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

elif activeTrainingBatch == TrainingBatch.CIFRA10:
    # storing the data-set
    cifar10.data_path = "C:/Users/Milan/Documents/Data/"
    cifar10.maybe_download_and_extract()
    class_names = cifar10.load_class_names()
