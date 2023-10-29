from tensorflow.keras.utils import Sequence
import math
import random
import numpy as np
from audiomentations import Compose, Normalize, Gain    
        
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
        
    
    def getClip(self, fnames):
        while True:
            f = random.choice(fnames)
            x = np.load(f)
            if(x.shape[0]==1_000_000):
                idx = random.choice(list(range(0, x.shape[0]-self.N) ))
                clip1 = x[idx:idx+self.N]
                idx = random.choice(list(range(0, x.shape[0]-self.N) ))
                clip2 = x[idx:idx + self.N]
                #clip1 = self.transform(clip1, sample_rate=self.SR)
                #clip2 = self.transform(clip2, sample_rate=self.SR)
                clip1 = self.rescale(clip1)
                clip2 = self.rescale(clip2)
                didx = random.choice(list(range(clip1.shape[0]-self.DROPOUT)))
                clip1[didx:didx+self.DROPOUT]=0
                didx = random.choice(list(range(clip2.shape[0]-self.DROPOUT)))
                clip2[didx:didx+self.DROPOUT]=0
                
                return clip1, clip2
            else:
                pass
 
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