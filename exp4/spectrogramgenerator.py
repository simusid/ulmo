from tensorflow.keras.utils import Sequence
import math
import random
import numpy as np
from audiomentations import Compose, Normalize, Gain    
import scipy 


class MyGenerator(Sequence):
    def __init__(self, fnames, batch_size=64, N=2048, dropout=500, sr=96000):
        self.fnames = fnames
        self.batch_size= 64
        self.N = N
        self.SR = sr
        self.DROPOUT = dropout
        self.transform = Compose([Normalize( p=1.0 )])
        
    def __len__(self):
        return math.ceil(len(self.fnames) / self.batch_size)
    
    def rescale(self, x):
        return -1 + 2 * (x - x.min()) / (x.max() - x.min())
        
    
    # returns the spectrogram of the time series
    # as a np array
    def spectrogram(self, x, nperseg=1024):
        _,_,y = scipy.signal.stft(x, nperseg=nperseg)
        return np.log(np.abs(y))
    
    
 
    def __getitem__(self, idx):
        x_train = []
        y_train = []
        for i in range(self.batch_size):
            a, ca = self.getClip(self.fnames)
            b, cb = self.getClip(self.fnames)
    
            choice = random.choice([0,1])
            y_train.append(choice)
             
            if(choice==0):
                #source is A
                clip = ca
            else:
                clip = cb
            assert len(clip)==self.N
            x_train.append(np.hstack([a,b,clip]))
    
        x_train = np.array(x_train)
        x_train = np.expand_dims(x_train,-1)  # add for conv network
        y_train = np.array(y_train)
        return x_train, y_train   