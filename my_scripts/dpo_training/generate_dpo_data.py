from itertools import product
from pathlib import Path
from uuid import uuid1

import click
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import tqdm

from TTS.api import TTS

project_dir = Path(__file__).resolve().parents[2]
default_prompt_path = project_dir / 'data' / 'dpo_texts_samples' / 'llama.csv'
default_speakers_dir = project_dir / 'data' / 'speakers'
default_model_path = project_dir / 'checkpoints' / 'last-lg'
default_output_path = project_dir / 'data' / 'dpo_dataset'


@click.command()
@click.option('--prompt-csv', default=default_prompt_path, help='Path to prompt CSV file')
@click.option('--sample-size', default=5, help='Number of samples to generate')
@click.option('--speakers-dir', default=default_speakers_dir, help='Path to speakers directory')
@click.option('--output-folder', default=default_output_path, help='Name of output folder')
def generate_tts(prompt_csv, sample_size, speakers_dir, output_folder):
    prompts = pd.read_csv(prompt_csv, index_col=0).iloc[:10, 0]
    model = TTS(
        model_name='xtts_v2',
        model_path=default_model_path,
        config_path=default_model_path / 'config.json',
    ).to('cuda')

    speakers_path = Path(speakers_dir)
    speakers = list(speakers_path.glob('*.wav'))

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'wav').mkdir(parents=True, exist_ok=True)

    metadata = []

    for (id_, prompt), speaker, gen_i in tqdm.tqdm(
            product(
                prompts.items(),
                speakers,
                range(sample_size)
            ),
            total=len(prompts) * len(speakers) * sample_size
    ):
        speaker_name = speaker.name.split('.')[0]

        # model.tts_to_file(
        #     text=prompt['prompt'],
        #     speaker_wav=speaker,
        #     language="en",
        #     file_path=output_path / f"({speaker_name})_{id_}_{gen_i}.wav"
        # )

        wav, gpt_codes = model.tts(
            text=prompt,
            speaker_wav=speaker,
            language="en",
        )
        output_audio_name = f'{str(uuid1())}.wav'

        sf.write(
            output_path/'wav'/output_audio_name,
            wav,
            samplerate=model.config.audio["output_sample_rate"],
        )

        metadata.append(
            {
                'gen_id': str(gen_i),
                'audio_id': output_audio_name,
                'speaker_id': speaker_name,
                'prompt_id': id_,
                'text': prompt,
                'gpt_codes': np.array(gpt_codes).flatten(),
            }
        )

    metadata = pd.DataFrame(metadata)
    metadata.to_parquet(output_path/'samples_data.parquet', engine='pyarrow')


if __name__ == '__main__':
    generate_tts()
