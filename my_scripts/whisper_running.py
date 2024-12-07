import glob
import os
from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# Initialize the Whisper processor and model
model_name = "openai/whisper-medium"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    chunk_length_s=30,
    device='cuda',
)

# Directory containing audio files
dataset_path = Path('../data/keithito_lj_speech')
metadata = pd.read_csv(dataset_path / 'metadata.csv', index_col=0)
metadata.set_index('audio_id', inplace=True)
train_metadata = pd.read_csv(dataset_path / 'train_metadata.csv')
test_metadata = pd.read_csv(dataset_path / 'test_metadata.csv')

whisper_transcription = {}

# Loop through all files in the directory
for audio_id, row in tqdm(metadata.iterrows()):
    file_path = dataset_path / 'wavs' / (audio_id + '.wav')
    wave, _ = librosa.load(file_path, sr=16000)
    transcription = pipe(wave)
    whisper_transcription[audio_id] = transcription['text']

metadata['whisper_transcription'] = pd.Series(whisper_transcription)
train_metadata = metadata.loc[metadata.index.intersection(train_metadata['audio_id'])]
test_metadata = metadata.loc[metadata.index.intersection(test_metadata['audio_id'])]

metadata.to_csv(dataset_path / 'metadata.csv')
train_metadata.to_csv(dataset_path / 'train_metadata.csv', index=False)
test_metadata.to_csv(dataset_path / 'test_metadata.csv', index=False)
