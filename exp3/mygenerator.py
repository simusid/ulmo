from tensorflow.keras.utils import Sequence
import math
import random
import numpy as np

from audiomentations import Compose, Normalize, Gain
 
transform1 = Compose([Normalize(p=1.0)])
transform2 = Compose([ Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0)])


def getbrokenfile(fnames):
    while True:
        f = random.choice(fnames)
        x = np.load(f)
        if(x.shape[0]==1_000_000):
            return x
        else:
            pass
        
        
        
class MyGenerator(Sequence):
    def __init__(self, fnames, batch_size=64, N=2048):
        self.fnames = fnames
        self.batch_size= 64
        self.N = N
        
    def __len__(self):
        return math.ceil(len(self.fnames) / self.batch_size)
    
    def __getitem__(self, idx):
        x_train = []
        y_train = []
        for i in range(self.batch_size):
            a = getbrokenfile(self.fnames)
            #a = transform2(transform1(a, sample_rate=96000), sample_rate=96000)
            a =  transform1(a, sample_rate=96000) 
            idx = random.choice(list(range(0, a.shape[0]-self.N) ))
            aa = a[idx:idx+self.N]
            b = getbrokenfile(self.fnames)
            #b = transform2(transform1(b, sample_rate=96000), sample_rate=96000)
            b =  transform1(b, sample_rate=96000) 
 
            idx = random.choice(list(range(0, b.shape[0]-self.N )))
            bb = b[idx:idx+self.N]
            
            choice = random.choice([0,1])
            y_train.append(choice)
            
            # now pick unknown
            idx = random.choice(list(range(0, a.shape[0]-self.N )))
            if(choice==0):
                #source is A
                clip = a[idx:idx+self.N]
            else:
                clip = b[idx:idx+self.N]
            assert len(clip)==self.N
            clip=  transform1(clip, sample_rate=96000) 
 
            x_train.append(np.hstack([aa,bb,clip]))
        x_train = np.array(x_train)
        x_train = np.expand_dims(x_train,-1)  # add for conv network
        y_train = np.array(y_train)
        return x_train, y_train   