import click
from itertools import product
from pathlib import Path

import pandas as pd
import tqdm

from TTS.api import TTS


@click.command()
@click.option('--prompt-csv', default='../data/my_tests/prompts.csv', help='Path to prompt CSV file')
@click.option('--sample-size', default=10, help='Number of samples to generate')
@click.option('--speakers-dir', default='../data/speakers/', help='Path to speakers directory')
@click.option('--output-folder', default='../data/my_tests/audio/respeacher_tts', help='Name of output folder')
def generate_tts(prompt_csv, sample_size, speakers_dir, output_folder):
    prompts = pd.read_csv(prompt_csv, index_col=0)
    model = TTS(model_path='../checkpoints/respeacher/', config_path='../checkpoints/respeacher/config.json').to('cuda')

    speakers_path = Path(speakers_dir)
    speakers = list(speakers_path.glob('*.wav'))

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for (id_, prompt), speaker, gen_i in tqdm.tqdm(
            product(
                prompts.iterrows(),
                speakers,
                range(sample_size)
            ),
            total=len(prompts) * len(speakers) * sample_size
    ):
        speaker_name = speaker.name.split('.')[0]

        model.tts_to_file(
            text=prompt['prompt'],
            speaker_wav=speaker,
            language="en",
            file_path=output_path / f"({speaker_name})_{id_}_{gen_i}.wav"
        )


if __name__ == '__main__':
    generate_tts()
