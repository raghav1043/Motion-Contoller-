#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:10:56 2019

@author: raghav
"""
from utils import *
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


def model_architecture():
    '''
    The network should accept a 96x96 grayscale image as input, and it should output a vector with 30 entries,
    corresponding to the predicted (horizontal and vertical) locations of 15 facial keypoints.
    '''
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(30))

    return model;

def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=1, validation_split=0.2)

def save_model(model, fileName):
    model.save(fileName + '.h5')

def load_cnn_model(fileName):
    return load_model(fileName + '.h5')

def load_cnn_model(weights_path):
   model = model_architecture()
   model.load_weights(weights_path)

X_train, y_train = train()
model = model_architecture()

compile_model(model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Training the model
history=train_model(model, X_train, y_train)

# Saving the model
save_model(model, 'model')
