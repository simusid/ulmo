"""
This is the first attempt to pretrain wav2vec2 using local data.    This requires
a local datasource, not hosted on hf-hub.

Use the modified version of the wav2vec2 pretraining script that uses
load_from_disk instead of a named source
"""

from glob import glob
from datasets import Dataset, DatasetDict , load_dataset, load_from_disk, Audio
import librosa 
from uuid import uuid4
from scipy.io import wavfile
from tqdm import tqdm 
import shutil 
import os
import random

N = 12500 # number of files
print("processing {} files".format(N))

SECONDS = 10  # wav2vec2 preprocessor has min/max limits 
RAW_WAV = "hammer_wavs"   # output directory for segments
NEW_DATASET = "/mnt/md0/HAMMER_DATASET"
GENRAW = True

fnames = glob("/mnt/md0/hammersounds/*wav")
random.shuffle(fnames) 
assert len(fnames)>0
if(GENRAW==True):
	try:
		shutil.rmtree(RAW_WAV)
	except:
		pass

	os.mkdir(RAW_WAV)
	os.mkdir(f"{RAW_WAV}/clean")
	os.mkdir(f"{RAW_WAV}/clean/train")
	os.mkdir(f"{RAW_WAV}/clean/test")

	for f in tqdm(fnames[:N]):
		x, sr = librosa.load(f, sr=16000)
		for idx in range(0, x.shape[0]-16000*SECONDS, 8000*SECONDS):
			y = x[idx:idx+16000*SECONDS]
			if(random.random()<0.9):
				dest ="train"
			else:
				dest="test"
			fn = f"{RAW_WAV}/clean/{dest}/{str(uuid4())}.wav"
			wavfile.write(fn, 16000, y)

	print("done exporting")
fnames = glob(RAW_WAV+"/clean/train/*wav")
assert len(fnames)>0
print('creating dataset of only train data')
audio_dataset = Dataset.from_dict({"file":fnames, "audio":fnames}).cast_column("audio", Audio())
audio_dataset.save_to_disk(NEW_DATASET)
