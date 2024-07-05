import glob
import os

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import load_dataset

# Initialize the Whisper processor and model
model_name = "openai/whisper-medium"


# processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)


# Function to process an audio file and return transcription
def transcribe_audio(file_path):
    # Load audio file
    dataset = load_dataset(audio_directory, data_files={"audio": file_path}, split="train")

    # Preprocess audio
    inputs = processor(dataset["audio"][0]["array"], return_tensors="pt", sampling_rate=16000)

    # Generate transcription
    generated_ids = model.generate(inputs["input_features"], max_length=500)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return transcription[0]


pipe = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    chunk_length_s=30,
    device='cuda',
)

# Directory containing audio files
audio_directory = r"..\my_tests\audio"
files = glob.glob(os.path.join(audio_directory, "*.wav"))

# Loop through all files in the directory
for file_path in files:
    print(f"Processing file: {file_path}")
    name = file_path.split('\\')[-1].split('.')[0]
    wave, _ = librosa.load(file_path, sr=16000)
    transcription = pipe(wave)
    with open(r'..\my_tests\whisper_output.txt', 'a') as f:
        f.write(f"{name}. {transcription['text']}\n")
