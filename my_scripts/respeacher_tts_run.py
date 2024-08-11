import pandas as pd
import tqdm

from TTS.api import TTS

prompts = pd.read_csv('../my_tests/prompts.csv', index_col=0)
model = TTS(model_path='../checkpoints/respeacher/', config_path='../checkpoints/respeacher/config.json').to('cuda')

sample_size = 10

for id_, prompt in tqdm.tqdm(prompts.iterrows(), total=len(prompts)):
    for i in range(sample_size):
        model.tts_to_file(
            text=prompt['prompt'],
            speaker_wav=r"D:\Учеба\КПИ\Diploma\TTS\tests\data\ljspeech\wavs\LJ001-0003.wav",
            language="en",
            file_path=f"../my_tests/audio/respeacher_tts/{id_}_{i}.wav"
        )
