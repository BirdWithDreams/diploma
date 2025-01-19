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
default_prompt_path = project_dir / 'data' / 'dpo_dataset' / 'llama_shuffled.csv'
# default_speakers_dir = project_dir / 'data' / 'speakers'
default_model_path = project_dir / 'checkpoints' / 'last-lg'
default_output_path = project_dir / 'data' / 'dpo_dataset'
default_speakers_dir = default_output_path / 'wavs'

model_name = 'base_xtts_v2'
model_path = project_dir


@click.command()
@click.option('--prompt-csv', default=default_prompt_path, help='Path to prompt CSV file')
@click.option('--sample-size', default=10, help='Number of samples to generate')
@click.option('--speakers-dir', default=default_speakers_dir, help='Path to speakers directory')
@click.option('--output-folder', default=default_output_path, help='Name of output folder')
@click.option('--prompts-bounds', default='na,na', help='Name of output folder')
@click.option('--output-name', default='dpo_data.parquet', help='Name of output folder')
def generate_tts(prompt_csv, sample_size, speakers_dir, output_folder, prompts_bounds, output_name):
    prompts = pd.read_csv(prompt_csv, index_col=0)

    low, high = prompts_bounds.split(',')
    if low == 'na':
        low = 0
    else:
        low = int(low)

    if high == 'na':
        high = len(prompts)
    else:
        high = int(high)

    prompts = prompts.iloc[low:high, 0]

    if model_name == 'base_xtts_v2':
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
    else:
        model = TTS(
            model_name='xtts_v2',
            model_path=model_path.as_posix(),
            config_path=(model_path / 'config.json').as_posix(),
        ).to('cuda')

    speakers_path = Path(speakers_dir)
    speakers = list(speakers_path.glob('*.wav'))

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    # (output_path / 'wavs').mkdir(parents=True, exist_ok=True)

    metadata = []

    prompts_and_speakers = zip(
        prompts.sample(frac=1, random_state=42).items(),
        chain(*[repeat(speaker, 1500) for speaker in speakers])
    )

    for i, (((id_, prompt), speaker), gen_i) in tqdm.tqdm(
            enumerate(product(
                prompts_and_speakers,
                range(sample_size)
            )),
            total=len(prompts) * len(speakers[:1]) * sample_size
    ):
        try:
            speaker_name = speaker.name.split('.')[0]

            wav, gpt_codes = model.tts(
                text=prompt,
                speaker_wav=speaker,
                language="en",
            )

            wav = np.array(wav, dtype=np.float32)

            transcription = pipe(wav)
            transcription = transcription['text']

            text_metrics = compute_text_metrics(prompt, transcription)
            secs_metric, utmos_metric = compute_audio_metric((wav, 22050), speaker.name, default_output_path)

            gpt_codes = np.hstack(gpt_codes)

            metadata.append(
                {
                    'gen_id': str(gen_i),
                    'audio_id': speaker.name,
                    'speaker_id': speaker_name,
                    'prompt_id': id_,
                    'text': prompt,
                    'gpt_codes': np.array(gpt_codes).flatten(),
                } |
                text_metrics |
                {
                    'secs': secs_metric,
                    'utmos': utmos_metric,
                }
            )
        except Exception as e:
            logger.error(f'Exception {str(e)} was occurred while processing {speaker.name} with {prompt} prompt')
            traceback.print_exc()

        if (i + 1) % 100 == 0:
            metadata = pd.DataFrame(metadata)
            metadata.to_parquet(output_path / f'batch_{i}.parquet', engine='pyarrow')
            metadata = []

    metadata = pd.DataFrame(metadata)
    metadata.to_parquet(output_path / output_name, engine='pyarrow')


if __name__ == '__main__':
    generate_tts()
