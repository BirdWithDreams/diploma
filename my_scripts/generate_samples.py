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

logger.add("../logs/generate_samples_{time}.log", rotation="10 MB", level="INFO")


def generate_samples(
        model_name,
        test_file,
        speakers,
        text_col='text',
        dataset_path='../data/keithito_lj_speech',
        model_path='../checkpoints/good_dataset/',
        output_path='/workspace/Projects/diploma/data/samples'
):
    logger.info(f"Starting TTS sampling for model: {model_name}")
    model_path = Path(model_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    output_path = Path(output_path).resolve()

    if test_file.endswith(".csv"):
        test_df = pd.read_csv(dataset_path / test_file)
    elif test_file.endswith(".parquet"):
        test_df = pd.read_parquet(dataset_path / test_file)

    logger.info(f"Loaded test file with {len(test_df)} samples from {test_file}")

    logger.info(f"Loading TTS model from {model_path}")

    if 'base_xtts_v2' in model_name:
        logger.info('Use base XTTS model')
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
    else:
        logger.info('Use model from checkpoint')
        model = TTS(
            # model_name='xtts_v2',
            model_path=model_path.as_posix(),
            config_path=(model_path / 'config.json').as_posix(),
        ).to('cuda')
    logger.debug("TTS model loaded successfully")

    # speakers = speakers.split(',')
    for speaker in speakers:
        (output_path / model_name / speaker).mkdir(parents=True, exist_ok=True)

    test_df = test_df[test_df['speaker_id'].isin(speakers)]


    for speaker_id, group in tqdm.tqdm(test_df.groupby('speaker_id')):
        group = group.iloc[:40]
        for id_, (_, row) in enumerate(tqdm.tqdm(group.iterrows(), total=len(group))):
            try:
                speaker_wav = dataset_path / 'wavs' / row['audio_id']

                logger.debug(f"Generating TTS for {row['audio_id']} audio, speaker {speaker_id}")
                # wave, gpt_codes = model.tts(
                #     text=row[text_col],
                #     speaker_wav=speaker_wav.as_posix(),
                #     language="en",
                # )

                model.tts_to_file(
                    text=row[text_col],
                    speaker_wav=speaker_wav.as_posix(),
                    language="en",
                    file_path=str(output_path / model_name / speaker_id / f'gen_{id_}.wav')
                )

                # try:
                #     wave = np.array(wave, dtype=np.float32)
                # except ValueError:
                #     wave = sum(wave, start=[])
                #     wave = np.array(wave, dtype=np.float32)

                with open(output_path / model_name / speaker_id / f'text_{id_}.txt', "w") as f:
                    f.write(row[text_col])

                # sf.write(output_path / model_name / row['speaker_id'] / f'gen_{id_}.wav', wave, 22050)
                shutil.copyfile(speaker_wav, output_path / model_name / speaker_id / f'gt_{id_}.wav')

            except Exception as e:
                logger.error(f'Error {str(e)} occurred while computing metrics for {row["audio_id"]} audio.')
                logger.error(logger.error(traceback.format_exc()))



if __name__ == '__main__':
    logger.info("Script started")
    fire.Fire(generate_samples)
    logger.info("Script finished")
