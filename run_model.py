import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description="run inference on a model")
parser.add_argument("--size", help="size of input images. eg 100 is (100,100)")
parser.add_argument("--input", help="preprocessed datafile (*.npz)")
parser.add_argument("--model", help="model directory")
 
args = parser.parse_args()
size = int(args.size)
fname = args.input
mname = args.model

print("loading data")
data  = np.load(fname)
print("done loading data")
x_train = data['x_train']
print("done unpacking data")
del data 
#print("reshaping data")
#x_train = np.reshape(x_train, (x_train.shape[0], size,size,3))#
print('loading model')
model = load_model(mname)
print('done loading model')
print(model.predict(x_train))
     

