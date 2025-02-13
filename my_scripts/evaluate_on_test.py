from pathlib import Path
import traceback

import fire
import numpy as np
import pandas as pd
import tqdm
from loguru import logger

from TTS.api import TTS
from evaluate_tts import compute_text_metrics, compute_audio_metric, pipe

logger.add("../logs/evaluate_on_test_{time}.log", rotation="10 MB", level="INFO")


def eval_tts(
        model_name,
        test_file,
        text_col='text',
        sample_size=10,
        dataset_path='../data/keithito_lj_speech',
        model_path='../checkpoints/good_dataset/',
        output_name='metrics'
):
    logger.info(f"Starting TTS evaluation for model: {model_name}")
    model_path = Path(model_path).resolve()
    dataset_path = Path(dataset_path).resolve()

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

    metrics = []
    for _, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            speaker_name = row['speaker_id']
            speaker_wav = dataset_path / 'wavs' / row['audio_id']

            logger.debug(f"Generating TTS for {row['audio_id']} audio, speaker {speaker_name}")
            wave, gpt_codes = model.tts(
                text=row[text_col],
                speaker_wav=speaker_wav.as_posix(),
                language="en",
            )

            try:
                wave = np.array(wave, dtype=np.float32)
            except ValueError:
                wave = sum(wave, start=[])
                wave = np.array(wave, dtype=np.float32)

            logger.debug("Transcribing generated audio")
            transcription = pipe(wave, generate_kwargs={"language": "english", 'return_timestamps': True})['text']

            logger.debug("Computing metrics")
            text_metrics = compute_text_metrics(row[text_col], transcription)
            secs_metric, utmos_metric = compute_audio_metric((wave, 22050), row['audio_id'], dataset_path)

            row = {
                'audio_id': row['audio_id'],
                'speaker_id': speaker_name,
                'model_name': model_name+'-test',
            }
            row |= text_metrics
            row |= {
                'secs': secs_metric,
                'utmos': utmos_metric,
            }

            row |= {
                'text': row['audio_id'],
                'transcription': transcription,
            }

            metrics.append(row)
            logger.debug(f"Metrics for current iteration: {row}")
        except Exception as e:
            logger.error(f'Error {str(e)} occurred while computing metrics for {row["audio_id"]} audio.')
            logger.error(logger.error(traceback.format_exc()))

    metrics_df = pd.DataFrame(metrics)
    output_path = dataset_path / 'metrics' / f'{output_name}.csv'
    logger.info(f"Saving metrics to {output_path}")
    metrics_df.to_csv(output_path, index=False)
    logger.info("Evaluation completed successfully")


if __name__ == '__main__':
    logger.info("Script started")
    fire.Fire(eval_tts)
    logger.info("Script finished")
