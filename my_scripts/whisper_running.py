import glob
import os
from pathlib import Path

import librosa
import pandas as pd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, WhisperModel, AutoModelForSpeechSeq2Seq, AutoProcessor

# Initialize the Whisper processor and model
model_name = "openai/whisper-medium"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_name)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Directory containing audio files
path_vctk = Path('/workspace/Projects/diploma/data/VCTK-Corpus')

# df = pd.read_csv(path_vctk / 'metadata.csv')
#
# res = []
# batch_size = 128
# audios = df['audio_id'].tolist()
# audios = [str(path_vctk / 'wavs' / audio) for audio in audios]
# transcripts = pipe(audios, batch_size=batch_size, generate_kwargs={"language": "english"})
# transcripts = [trans['text'] for trans in transcripts]
#
# df['whisper_transcription'] = transcripts
#
# df.to_csv(path_vctk / 'metadata.csv')

audio = list((path_vctk / 'wavs').glob('*.wav'))[:2]
audio = [librosa.load(a)[0] for a in audio]
t = pipe(audio, batch_size=2, generate_kwargs={"language": "english"})
pass