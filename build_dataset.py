"""
This is the first attempt to pretrain wav2vec2 using local data.    This requires
a local datasource, not hosted on hf-hub.

Use the modified version of the wav2vec2 pretraining script that uses
load_from_disk instead of a named source
"""

from glob import glob
from datasets import Dataset , load_dataset, load_from_disk, Audio

RAW_WAV = "raw_wav"
DEST_DATASET='audio_dataset'
fnames = glob(f"{RAW_WAV}/*")
audio_dataset = Dataset.from_dict({"audio":fnames}).cast_column("audio", Audio())
audio_dataset[0]["audio"]
audio_dataset.save_to_disk(f"DEST_DATASET")