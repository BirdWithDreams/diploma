import torch
from tqdm import tqdm
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="sample.m4a", language="en")
# Text to speech to a file
with open('../my_tests/prompts', 'r') as f:
    prompts = f.readlines()

for prompt in tqdm(prompts[50:]):
    number = prompt.split(' ', maxsplit=1)[0]
    prompt = prompt.split(' ', maxsplit=1)[1]

    tts.tts_to_file(
        text=prompt,
        speaker_wav=r"D:\Учеба\КПИ\Diploma\TTS\tests\data\ljspeech\wavs\LJ001-0003.wav",
        language="en",
        file_path=f"my_tests/audio/{number}.wav"
    )
