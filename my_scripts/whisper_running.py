import glob
import os

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import load_dataset

# Initialize the Whisper processor and model
model_name = "openai/whisper-medium"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    chunk_length_s=30,
    device='cuda',
)

# Directory containing audio files
audio_directory = r"../data/my_tests/audio/good_fn_sample"
files = glob.glob(os.path.join(audio_directory, "*.wav"))

# Loop through all files in the directory
for file_path in files:
    print(f"Processing file: {file_path}")
    name = file_path.split('\\')[-1].split('.')[0]
    wave, _ = librosa.load(file_path, sr=16000)
    transcription = pipe(wave)
    with open(r'../data/my_tests/audio//whisper_good_fn_sample.txt', 'a') as f:
        f.write(f"{name}. {transcription['text']}\n")
