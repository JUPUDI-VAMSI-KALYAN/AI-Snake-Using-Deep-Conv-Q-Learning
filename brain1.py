# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:19:08 2023

@author: 0036YD744
"""

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam


class Brain():
    
    def __init__(self, input_shape, lr = 0.005):
        self.input_shape = input_shape
        self.learningRate = lr
        self.numoutput = 4
        
        self.model = Sequential()
        self.model.add(Conv2D(32,(3,3),activation='relu', input_shape = self.input_shape))
        
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Conv2D(64,(2,2),activation='relu'))
        
        self.model.add(Flatten())
        self.model.add(Dense(256,activation='relu'))
        self.model.add(Dense(self.numoutput))
        self.model.compile(optimizer = Adam(lr=self.learningRate), loss = 'mean_squared_error')
    

    def loadmodel(self,filepath):
        self.model = load_model(filepath)
        return self.model