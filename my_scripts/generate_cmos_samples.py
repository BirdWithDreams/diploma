method_pairs = {
    '1-4': [
        ('L1', {'model_name': 'lg-human-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('L4', {'model_name': 'base-lg-dpo-augmented-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('V1', {'model_name': 'vctk_last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
        ('V4', {'model_name': 'base-vctk-dpo-augmented-last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'})
    ],
    '2-4': [
        ('L2', {'model_name': 'lg-asr', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('L4', {'model_name': 'base-lg-dpo-augmented-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('V2', {'model_name': 'vctk-asr', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
        ('V4', {'model_name': 'base-vctk-dpo-augmented-last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'})
    ],
    '1-6': [
        ('L1', {'model_name': 'lg-human-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('L6', {'model_name': 'asr-lg-dpo-augmented-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('V1', {'model_name': 'vctk_last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
        ('V6', {'model_name': 'asr-vctk-dpo-augmented-last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'})

    ],
    '2-6': [
        ('L2', {'model_name': 'lg-asr', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('L6', {'model_name': 'asr-lg-dpo-augmented-last', 'dataset_path': '../data/keithito_lj_speech', 'dataset_name': 'lj_speech'}),
        ('V2', {'model_name': 'vctk-asr', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'}),
        ('V6', {'model_name': 'asr-vctk-dpo-augmented-last', 'dataset_path': '../data/VCTK-Corpus', 'dataset_name': 'vctk'})
    ],
}

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

logger.add("../logs/generate_cmos_samples_{time}.log", rotation="10 MB", level="INFO")


def generate_cmos_samples(cmos_speakers_file='../data/cmos_speakers.csv', models_folder='../checkpoints/finale_models', output_folder='../data/cmos_data'):
    cmos_speakers = pd.read_csv(cmos_speakers_file)
    groups = cmos_speakers.groupby(['GENDER', 'ID', 'speaker_id'])

    output_folder = Path(output_folder)
    models_folder = Path(models_folder)

    cmos_metadata = []

    for method_pair, models in method_pairs.items():
        for model_id, model_params in models:
            model_name = model_params['model_name']
            dataset_path = Path(model_params['dataset_path'])
            dataset_name = model_params['dataset_name']

            model_path = models_folder / model_name
            model = TTS(
                model_path=model_path.as_posix(),
                config_path=(model_path / 'config.json').as_posix(),
            ).to('cuda')

            # model.synthesizer.tts_model.gpt.init_gpt_for_inference(kv_cache=model.args.kv_cache, use_deepspeed=False)
            # model.synthesizer.tts_model.gpt.eval()

            for (gender, id_, speaker_id), samples in groups:
                if samples['dataset'].iloc[0] == 'lj_speech':
                    pass
                if samples['dataset'].iloc[0] != dataset_name:
                    continue

                speaker_wav = dataset_path / 'wavs'

                for sample_number, (audio_id, text) in enumerate(zip(samples['audio_id'], samples['text']), start=1):
                    speaker_sample_folder = output_folder / method_pair / speaker_id / f'sample_{sample_number}'
                    speaker_sample_folder.mkdir(parents=True, exist_ok=True)
                    model.tts_to_file(
                        text=text,
                        speaker_wav=(speaker_wav / audio_id).as_posix(),
                        language="en",
                        file_path=str(speaker_sample_folder / f'gen_{model_id}.wav')
                    )

                    shutil.copyfile(speaker_wav / audio_id, speaker_sample_folder / 'ref.wav')
                    with open(speaker_sample_folder / 'text.txt', 'w') as f:
                        f.write(text)

                    cmos_metadata.append({
                        'method_pair': method_pair,
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
    cmos_metadata.to_csv(output_folder / 'cmos_metadata.csv', index=False)


if __name__ == '__main__':
    logger.info("Script started")
    fire.Fire(generate_cmos_samples)
    logger.info("Script finished")

# method pair
#     speaker
#         sample_1
#             ref.wav
#             text.txt
#             gen_1.wav
#             gen_2.wav
#         sample_2
#             ref.wav
#             text.txt
#             gen_1.wav
#             gen_2.wav
#         ...


# speaker ref_audio text
