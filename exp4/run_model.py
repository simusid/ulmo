import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from glob import glob

import math
import random
from mygenerator import MyGenerator

 
N = 4096  # length of 
DROPOUT = 500
SR = 96000
ROOT="/tmp/foo/sancsound/"

fnames = glob(ROOT+ "clips/train/*npy")
testnames = glob(ROOT+ 'clips/test/*npy')
 
mygen = MyGenerator( fnames, N=N, dropout=DROPOUT, sr=SR)
testgen = MyGenerator( testnames, N=N, dropout=DROPOUT, sr=SR )

checkpoint =ModelCheckpoint(ROOT + "best_{epoch}", save_best=False)
 

def buildABConvModel(N):
    model = Sequential()
    model.add(Conv1D(16,3, input_shape=(N*3,1), activation='relu'))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(64,3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(64,3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(128,3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(128,3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='sigmoid'))
    return model
   

adam = Adam(learning_rate=0.0003)
model = buildABConvModel(N)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
model.summary()

model.fit(mygen, validation_data=testgen, 
                 validation_steps=10,  
                 batch_size=256, epochs=50, callbacks=[checkpoint] )
