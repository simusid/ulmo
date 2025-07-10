#!/usr/bin/env python
"""
A complete example to pretrain a Wav2Vec2 model from scratch using Hugging Face Transformers.

Assumptions:
  • Your original audio files are stored in /data/wavs and are in .wav format at 48kHz.
  • It is more efficient (and common) to train on short segments (e.g. 5 sec) than the full 20 minutes.
  • We’ll resample to 16kHz (adjust if needed) and use a custom Dataset that yields a random 5-second crop.
  • We create a feature extractor and a data-collator to prepare batches.
  • We use Wav2Vec2ForPreTraining (which automatically applies the mask and contrastive loss) and a custom configuration.
  
NOTE: Training a model from scratch is computationally expensive and may require substantial data.
"""

import os
import random
import librosa
import numpy as np
import torch

from transformers import (
    Wav2Vec2ForPreTraining,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
    Trainer,
    TrainingArguments
)

# -------------------------------------
# 1. Define a custom Dataset
# -------------------------------------
class WavDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, target_sample_rate=16000, segment_length=5):
        """
        audio_dir: directory with wav files.
        target_sample_rate: the sample rate to use (e.g. 16000 Hz).
        segment_length: length (in seconds) of the random segment used for training.
        """
        # List all .wav files in the directory
        self.file_paths = [
            os.path.join(audio_dir, fname)
            for fname in os.listdir(audio_dir)
            if fname.endswith(".mp3")
        ]
        self.target_sample_rate = target_sample_rate
        self.segment_length = segment_length  # in seconds
        self.segment_samples = self.segment_length * self.target_sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load the wav file and simultaneously resample to target_sample_rate.
        # librosa.load will convert the sample rate.
        audio, _ = librosa.load(file_path, sr=self.target_sample_rate)
        # If the audio is longer than the desired segment, randomly crop
        if len(audio) > self.segment_samples:
            max_start = len(audio) - self.segment_samples
            start = random.randint(0, max_start)
            audio = audio[start : start + self.segment_samples]
        else:
            # If the file is too short, pad with zeros at the end
            audio = np.pad(audio, (0, self.segment_samples - len(audio)), mode="constant")

        # Return a dictionary; here the key "input_values" is what the model expects.
        return {"input_values": audio}

# -------------------------------------
# 2. Instantiate the Dataset
# -------------------------------------
audio_directory = "/home/gary/Downloads/birdsong-recognition/train_audio/amerob/"  # Change to your path
dataset = WavDataset(audio_directory, target_sample_rate=16000, segment_length=5)

# -------------------------------------
# 3. Define the Feature Extractor
# -------------------------------------
# The feature extractor ensures that the raw waveform is normalized and padded correctly.
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

# -------------------------------------
# 4. Create a Data Collator
# -------------------------------------
def data_collator(batch):
    """
    Pads a list of examples to the same length and returns a batch dictionary.
    Each example is expected to have an "input_values" key.
    """
    # Extract list of np.array for "input_values"
    input_list = [example["input_values"] for example in batch]
    # Use the feature extractor’s pad method.
    batch = feature_extractor(input_list, padding=True, return_tensors="pt")
    return batch

# -------------------------------------
# 5. Configure and Instantiate the Model
# -------------------------------------
# Here we define a configuration for pretraining. You can adjust parameters as needed.
config = Wav2Vec2Config(
    # Model architecture parameters (example values)
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    # Convolutional feature extractor (example settings)
    conv_dim=(512, 512, 512, 512, 512, 512, 512),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    # Pretraining masking hyperparameters
    mask_time_prob=0.065,       # probability to mask a frame in time dimension
    mask_time_length=10,        # length of the masked time span (in time steps)
    mask_feature_prob=0.0,      # some setups also mask feature dim (set to 0 to disable)
    mask_feature_length=0,
    # You can add further parameters (like model dropout, layer norm epsilon, etc.)
    # Note: The configuration for wav2vec2 pretraining may expect several additional parameters.
)

# Initialize the model for pretraining.
model = Wav2Vec2ForPreTraining(config)

# -------------------------------------
# 6. Define Training Arguments
# -------------------------------------
# Adjust batch sizes, learning rate, and number of epochs according to your needs and hardware.
training_args = TrainingArguments(
    output_dir="./wav2vec2_pretraining",
    per_device_train_batch_size=4,       # adjust based on your GPU memory
    gradient_accumulation_steps=4,
    num_train_epochs=10,                   # set the number of epochs (this is an example)
    learning_rate=5e-4,
    warmup_steps=500,
    logging_steps=50,
    save_steps=500,
    fp16=True,                           # enable mixed precision if supported
    evaluation_strategy="no",            # change as needed; here we don’t evaluate during pretraining
)

# -------------------------------------
# 7. Instantiate the Trainer and Begin Training
# -------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

if __name__ == "__main__":
    trainer.train()
    # Save the final model (+ configuration and feature extractor)
    trainer.save_model(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)
    
    print("Finished pretraining!")
