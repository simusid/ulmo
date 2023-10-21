import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from umap import UMAP
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

fnames = glob("clips/train/*npy")
testnames = glob('clips/test/*npy')
len(fnames), len(testnames)

def getbrokenfile(fnames):
    while True:
        f = random.choice(fnames)
        x = np.load(f)
        if(x.shape[0]==1_000_000):
            return x
        else:
            pass

import math
import random

N = 1024
class MyGenerator(Sequence):
    def __init__(self, fnames, batch_size=64):
        self.fnames = fnames
        self.batch_size= 64
        
    def __len__(self):
        return math.ceil(len(self.fnames) / self.batch_size)
    
    def __getitem__(self, idx):
        x_train = []
        y_train = []
        for i in range(self.batch_size):
            a = getbrokenfile(self.fnames)
            a = a-a.min()
            a = a/a.max()
            idx = random.choice(list(range(0, a.shape[0]-N) ))
            aa = a[idx:idx+N]
            b = getbrokenfile(self.fnames)
            b = b-b.min()
            b = b/b.max()
 
            idx = random.choice(list(range(0, b.shape[0]-N )))
            bb = b[idx:idx+N]
            
            choice = random.choice([0,1])
            y_train.append(choice)
            idx = random.choice(list(range(0, a.shape[0]-N )))
             
            if(choice==0):
                #source is A
                clip = a[idx:idx+N]
            else:
                clip = b[idx:idx+N]
            assert len(clip)==N
             
            x_train.append(np.hstack([aa,bb,clip]))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train     
        
mygen = MyGenerator(fnames)
testgen = MyGenerator(testnames)
es = tf.keras.callbacks.EarlyStopping(patience=4)
 
checkpoint =ModelCheckpoint("best_models/cp_{epoch:02d}", save_best=False)
def buildABModel(N):
    # size of model is N
    model = Sequential()
    model.add(Dense(768, input_shape=(N*3,), activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='sigmoid'))
    return model
   
model = buildABModel(N)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

history = model.fit(mygen,validation_data=testgen, 
                   validation_steps=10,  
                    batch_size=256, epochs=50, callbacks=[es, checkpoint] )
with open("history.pkl", "wb") as fh:
    pkl.dump(history, fh)




