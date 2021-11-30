# simple example image preprocessing script
# use this as a template
from PIL import Image
from glob import glob
import argparse
import numpy as np

parser=argparse.ArgumentParser(description="example preprocessing script for images")
parser.add_argument("--size", help="size of resized images. eg 100 is (100,100)")
parser.add_argument("--input", help="input directory of files")
parser.add_argument("--output", help="output npz file name")
args = parser.parse_args()

size = int(args.size)
dirname = args.input 
output = args.output
x_train = []
y_train =[]

for f in glob(dirname+"/*"):
    img = Image.open(f).resize((size,size)).convert("RGB")
    img = np.array(img)/255.
    img = np.reshape(img, (size,size,3))
    x_train.append(img)

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0],size, size, 3))
y_train = np.array(y_train)
np.savez(f"{output}.npz", x_train=x_train)

 
