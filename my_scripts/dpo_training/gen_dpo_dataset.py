import traceback
from itertools import product, chain, repeat
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tqdm
from loguru import logger

from TTS.api import TTS
from evaluate_tts import compute_text_metrics, compute_audio_metric, pipe

logger.add("../../logs/generate_dpo_data.log", rotation="10 MB", level="INFO")

project_dir = Path(__file__).resolve().parents[2]
default_prompt_path = project_dir / 'data' / 'VCTK-Corpus'
# default_speakers_dir = project_dir / 'data' / 'speakers'
default_model_path = project_dir / 'checkpoints' / 'finale_models' / 'vctk-asr'
default_output_path = project_dir / 'data' / 'dpo_dataset' / 'vctk-asr'



@click.command()
@click.option('--model-path', default=default_model_path, help='Path to dataset')
@click.option('--dataset-path', default=default_prompt_path, help='Path to dataset')
@click.option('--sample-size', default=10, help='Number of samples to generate')
@click.option('--output-folder', default=default_output_path, help='Name of output folder')
@click.option('--prompts-bounds', default='na,na', help='Name of output folder')
@click.option('--batch-size', default=100, help='The size of batch to save')
def generate_tts(model_path, dataset_path, sample_size, output_folder, prompts_bounds, batch_size):
    model_path = Path(model_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    metadata = pd.read_csv(dataset_path / 'metadata.csv')

    low, high = prompts_bounds.split(',')
    if low == 'na':
        low = 0
    else:
        low = int(low)

    if high == 'na':
        high = len(metadata)
    else:
        high = int(high)

    metadata = metadata.iloc[low:high]

    model = TTS(
        # model_name='xtts_v2',
        model_path=model_path.as_posix(),
        config_path=(model_path / 'config.json').as_posix(),
    ).to('cuda')

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    # (output_path / 'wavs').mkdir(parents=True, exist_ok=True)

    dpo_samples = []
    wavs = []


    for i, ((id_, row), gen_i) in tqdm.tqdm(
            enumerate(product(
                metadata.iterrows(),
                range(sample_size)
            )),
            total=len(metadata) * sample_size
    ):
        try:
            speaker_name = row['speaker_id']

            if row['audio_id'].endswith('.wav'):
                audio_id = row['audio_id']
            else:
                audio_id = row['audio_id'] + '.wav'

            speaker = dataset_path / 'wavs' / audio_id
            prompt = row['whisper_transcription']

            wav, gpt_codes = model.tts(
                text=prompt,
                speaker_wav=speaker,
                language="en",
            )

            wav = np.array(wav, dtype=np.float32)
            wavs.append(wav)

            secs_metric, utmos_metric = compute_audio_metric((wav, 22050), speaker.name, dataset_path)

            gpt_codes = np.hstack(gpt_codes)

            dpo_samples.append(
                {
                    'gen_id': str(gen_i),
                    'audio_id': speaker.name,
                    'speaker_id': speaker_name,
                    'prompt_id': id_,
                    'text': prompt,
                    'gpt_codes': np.array(gpt_codes).flatten(),
                    'secs': secs_metric,
                    'utmos': utmos_metric,
                }
            )
        except Exception as e:
            logger.error(f'Exception {str(e)} was occurred while processing {speaker.name} with {prompt} prompt')
            traceback.print_exc()

        if (i + 1) % batch_size == 0:
            try:
                transcriptions = pipe(wavs, batch_size=50, generate_kwargs={"language": "english"})
                transcriptions = [transcription['text'] for transcription in transcriptions]

                prompts = [row['text'] for row in dpo_samples]

                text_metrics = [compute_text_metrics(prompt, transcription) for prompt, transcription in zip(prompts, transcriptions)]
                text_metrics = pd.DataFrame(text_metrics)

                dpo_samples = pd.DataFrame(dpo_samples)
                dpo_samples = pd.concat([dpo_samples, text_metrics], axis=1)
                dpo_samples.to_parquet(output_path / f'batch_{low+i}.parquet', engine='pyarrow')


            except Exception as e:
                print(f"Error {str(e)} occure while processing {i}-th batch")
            finally:
                dpo_samples = []
                wavs = []

    dpo_samples = pd.DataFrame(dpo_samples)
    dpo_samples.to_parquet(output_path / f'batch_{low+i}.parquet', engine='pyarrow')


if __name__ == '__main__':
    generate_tts()
