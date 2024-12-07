from itertools import product
from pathlib import Path

import pandas as pd
import torch
from matplotlib.patches import PathPatch
from tqdm import tqdm
from TTS.api import TTS

prompts = pd.read_csv('../data/my_tests/prompts.csv')

sentences = prompts.iloc[[10, 51, 53]]['prompt'].tolist()

models = {
    # 'best-lg': '../checkpoints/lg_best',
    # 'last-lg': '../checkpoints/lg_last',
    # 'best-vp': '../checkpoints/vox_best',
    # 'last-vp': '../checkpoints/vox_last',
    # 'base_xtts_v2': 'base_xtts_v2',
    # 'best-vctk': '../checkpoints/vctk_best',
    'last-vctk': '../checkpoints/vctk_last',
}

datasets = [
    # '../data/facebook_voxpopuli',
    # '../data/keithito_lj_speech',
    '../data/VCTK-Corpus/',
]

output_dir = Path('../data/samples')

for model_name, model_path in models.items():
    if model_name =='base_xtts_v2':
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
    else:
        model_path = Path(model_path)
        model = TTS(
            model_name='xtts_v2',
            model_path=model_path.as_posix(),
            config_path=(model_path / 'config.json').as_posix(),
        ).to('cuda')

    for (sentence_id, sentence), dataset in tqdm(product(enumerate(sentences), datasets), total=len(sentences)*len(datasets)):
        dataset_speakers = Path(dataset) / 'speakers_to_sample.csv'
        dataset_speakers = pd.read_csv(dataset_speakers)
        for _, row in dataset_speakers.iterrows():
            speaker_wav = Path(dataset) / 'wavs' / row['audio_id']
            output_file = output_dir / f'{model_name}-{row["speaker_id"]}-{row["ACCENTS"]}-{sentence_id}.wav'
            model.tts_to_file(
                text=sentence,
                speaker_wav=speaker_wav,
                language="en",
                file_path=output_file,
            )