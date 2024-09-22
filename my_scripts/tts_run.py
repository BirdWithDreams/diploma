import pandas as pd
import torch
from tqdm import tqdm
from TTS.api import TTS

prompts = pd.read_csv('../data/my_tests/prompts.csv', index_col=0)
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# sample_size = 10
#
# for id_, prompt in tqdm(prompts.iterrows(), total=len(prompts)):
#     for i in range(sample_size):
#         tts.tts_to_file(
#             text=prompt['prompt'],
#             speaker_wav=r"/home/azhuravlov/Projects/diploma/data/speakers/LJ001-0001.wav",
#             language="en",
#             file_path=f"../data/my_tests/audio/xTTs/{id_}_{i}.wav"
#         )

tts.tts_to_file(
            text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            speaker_wav=r"/home/azhuravlov/Projects/diploma/data/speakers/LJ001-0001.wav",
            language="en",
            file_path=f"../data/my_tests/lj_sample.wav"
        )
