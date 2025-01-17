

from lam import *
from glob import glob
import matplotlib.pyplot as plt

fnames = glob("/Users/gary/Downloads/birdsong-recognition/train_audio/**/*mp3")
grams = MultiChannelGrams(fnames[0]).grams
plt.imshow(grams[0])
input("")