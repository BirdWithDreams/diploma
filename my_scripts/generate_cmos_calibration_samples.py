methods = [
        ('L1', {'model_name': 'lg-human-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('L2', {'model_name': 'lg-asr', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('L4', {'model_name': 'base-lg-dpo-augmented-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('L6', {'model_name': 'asr-lg-dpo-augmented-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),

        ('V1', {'model_name': 'vctk_last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
        ('V2', {'model_name': 'vctk-asr', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
        ('V4', {'model_name': 'base-vctk-dpo-augmented-last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
        ('V6', {'model_name': 'asr-vctk-dpo-augmented-last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
]

from pathlib import Path
import shutil
import traceback

import fire
import soundfile as sf
import numpy as np
import pandas as pd
import tqdm
from loguru import logger

from TTS.api import TTS

logger.add("../logs/generate_cmos_calibration_samples_{time}.log", rotation="10 MB", level="INFO")


def generate_cmos_calib_samples(cmos_calib_file='../data/cmos_calib_samples.csv', models_folder='../checkpoints/finale_models', output_folder='../data/cmos_calibration_data'):
    cmos_speakers = pd.read_csv(cmos_calib_file)

    output_folder = Path(output_folder)
    models_folder = Path(models_folder)

    cmos_metadata = []

    for model_id, model_params in methods:
        method = model_id[1]
        model_name = model_params['model_name']
        dataset_path = Path(model_params['dataset_path'])
        dataset_name = model_params['dataset_name']

        model_path = models_folder / model_name
        model = TTS(
            model_path=model_path.as_posix(),
            config_path=(model_path / 'config.json').as_posix(),
        ).to('cuda')

        for speaker_id, audio_id, text in cmos_speakers.itertuples(index=False):
            if speaker_id.startswith('p') and model_id.startswith('L'):
                continue

            if speaker_id == 'lj_speaker' and model_id.startswith('V'):
                continue

            speaker_wav = dataset_path / 'wavs'
            speaker_sample_folder = output_folder / method / speaker_id
            speaker_sample_folder.mkdir(parents=True, exist_ok=True)
            model.tts_to_file(
                text=text,
                speaker_wav=(speaker_wav / audio_id).as_posix(),
                language="en",
                file_path=str(speaker_sample_folder / f'gen.wav')
            )

            shutil.copyfile(speaker_wav / audio_id, speaker_sample_folder / 'gt.wav')
            with open(speaker_sample_folder / 'text.txt', 'w') as f:
                f.write(text)

            cmos_metadata.append({
                'method': method,
                'model_id': model_id,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'speaker_id': speaker_id,
                'audio_id': audio_id,
                'ref_wav': str((speaker_wav / audio_id).absolute()),
                'generated_wav': str((speaker_sample_folder / f'gen_{model_id}.wav').absolute()),
                'text': text,
            })

    cmos_metadata = pd.DataFrame(cmos_metadata)
    cmos_metadata.to_csv(output_folder / 'cmos_calibration_metadata.csv', index=False)


if __name__ == '__main__':
    logger.info("Script started")
    fire.Fire(generate_cmos_calib_samples)
    logger.info("Script finished")

# audio_files_calibration/
# └───<method_name_A>/  (e.g., model_baseline_calib)
# │   ├───<speaker_id_1>/
# │   │   ├───gt.wav
# │   │   ├───gen.wav
# │   │   └───text.txt
# │   └───<speaker_id_2>/
# │       └───...
# └───<method_name_B>/
#     └───...
