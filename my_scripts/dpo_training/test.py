import pandas as pd
from pathlib import Path


def group_scoring(group):
    # Sort values and reset index to get positions 1 to n
    sorted_group = group.sort_values()
    n = len(sorted_group)

    # Create scores from 1 to n (or 1 to 10 if you specifically need 10)
    scores = pd.Series(
        data=range(1, n + 1),
        index=sorted_group.index
    )
    return scores


# Read the data
df = pd.read_parquet('../../data/dpo_dataset/finale_samples.parquet')

# Apply ranking for each metric
df['cer_rank'] = df.groupby(['speaker_id', 'prompt_id'])['cer'].transform(group_scoring)
df['secs_rank'] = df.groupby(['speaker_id', 'prompt_id'])['secs'].transform(group_scoring)
df['utmos_rank'] = df.groupby(['speaker_id', 'prompt_id'])['utmos'].transform(group_scoring)

df.to_parquet('../../data/dpo_dataset/finale_samples.parquet')

# import traceback
# from itertools import product, chain, repeat
# from pathlib import Path
#
# import click
# import numpy as np
# import pandas as pd
# import tqdm
# from loguru import logger
#
# from TTS.api import TTS
# from evaluate_tts import compute_text_metrics, compute_audio_metric, pipe
#
# logger.add("../../logs/test.log", rotation="10 MB", level="INFO")
#
# project_dir = Path(__file__).resolve().parents[2]
# default_prompt_path = project_dir / 'data' / 'dpo_dataset' / 'llama_shuffled.csv'
# # default_speakers_dir = project_dir / 'data' / 'speakers'
# default_model_path = project_dir / 'checkpoints' / 'last-lg'
# default_output_path = project_dir / 'data' / 'dpo_dataset'
# default_speakers_dir = default_output_path / 'wavs'
#
# model_name = 'base_xtts_v2'
# model_path = project_dir
# sample_size = 10
#
# df = pd.read_parquet('../../data/dpo_dataset/finale_samples.parquet')
# prompts = df[df['speaker_id'] == 'LJ001-0002'][['prompt_id', 'text']].copy()
# del df
#
# prompts = prompts.drop_duplicates()
# prompts.set_index('prompt_id', inplace=True)
#
# low = 1500
# high = 3000
# prompts = prompts.sort_index().iloc[low:high, 0]
#
# if model_name == 'base_xtts_v2':
#     model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
# else:
#     model = TTS(
#         model_name='xtts_v2',
#         model_path=model_path.as_posix(),
#         config_path=(model_path / 'config.json').as_posix(),
#     ).to('cuda')
#
# speakers = [Path('../../data/dpo_dataset/wavs/p236_441.wav')]
#
# output_path = Path('extra_2')
# output_path.mkdir(parents=True, exist_ok=True)
# # (output_path / 'wavs').mkdir(parents=True, exist_ok=True)
#
# metadata = []
#
# prompts_and_speakers = zip(
#     prompts.sample(frac=1, random_state=42).items(),
#     chain(*[repeat(speaker, 1500) for speaker in speakers])
# )
#
# for i, (((id_, prompt), speaker), gen_i) in tqdm.tqdm(
#         enumerate(product(
#             prompts_and_speakers,
#             range(sample_size)
#         )),
#         total=len(prompts) * len(speakers[:1]) * sample_size
# ):
#     try:
#         # if gen_i >= 5:
#         #     speaker = random.choice(speakers)
#
#         speaker_name = speaker.name.split('.')[0]
#
#         wav, gpt_codes = model.tts(
#             text=prompt,
#             speaker_wav=speaker,
#             language="en",
#         )
#
#         wav = np.array(wav, dtype=np.float32)
#
#         transcription = pipe(wav)
#         transcription = transcription['text']
#
#         text_metrics = compute_text_metrics(prompt, transcription)
#         secs_metric, utmos_metric = compute_audio_metric((wav, 22050), speaker.name, default_output_path)
#
#         gpt_codes = np.hstack(gpt_codes)
#
#         metadata.append(
#             {
#                 'gen_id': str(gen_i),
#                 'audio_id': speaker.name,
#                 'speaker_id': speaker_name,
#                 'prompt_id': id_,
#                 'text': prompt,
#                 'gpt_codes': np.array(gpt_codes).flatten(),
#             } |
#             text_metrics |
#             {
#                 'secs': secs_metric,
#                 'utmos': utmos_metric,
#             }
#         )
#     except Exception as e:
#         logger.error(f'Exception {str(e)} was occurred while processing {speaker.name} with {prompt} prompt')
#         traceback.print_exc()
#
#     if (i + 1) % 100 == 0:
#         metadata = pd.DataFrame(metadata)
#         metadata.to_parquet(output_path / f'batch_{i}.parquet', engine='pyarrow')
#         metadata = []
#
# metadata = pd.DataFrame(metadata)
# metadata.to_parquet(output_path / 'last.parquet', engine='pyarrow')


# import os
# from itertools import count
#
# import click
# import pandas as pd
# import multiprocessing
# from typing import Tuple, List
#
# from gen_dpo_dataset import generate_tts
#
#
# def gen_bound_pairs(size: int, num_threads: int) -> List[Tuple[int, int]]:
#     """
#     Generate bound pairs for parallel processing.
#
#     Args:
#         size (int): Total size of the dataset
#         num_threads (int): Number of parallel threads/processes
#
#     Returns:
#         List of (start, end) tuples for dividing work
#     """
#     batch_size = size // num_threads + 1
#     bounds = list(range(0, size, batch_size)) + [size]
#     return list(zip(bounds[:-1], bounds[1:]))
#
#
# def run_parallel_generation(
#         num_processes: int,
#         batch_size: int
# ) -> None:
#     """
#     Run parallel generation across multiple datasets and models.
#
#     Args:
#         num_processes (int): Number of parallel processes
#         batch_size (int): Batch size for processing
#     """
#     # Define datasets and models
#     datasets = [
#         ['../../data/VCTK-Corpus', '../../data/VCTK-Corpus_gen'],
#         ['../../data/keithito_lj_speech', '../../data/keithito_lj_speech_gen'],
#     ]
#     models = [
#         'vctk-asr',
#         'lg-asr',
#     ]
#
#     # Compute dataset sizes
#     dataset_size = [
#         [len(pd.read_csv(d + '/metadata.csv')) for d in data]
#         for data in datasets
#     ]
#
#     # Generate bounds for parallel processing
#     bounds = [
#         [gen_bound_pairs(d_s, num_processes) for d_s in data_size]
#         for data_size in dataset_size
#     ]
#
#     # Prepare arguments for multiprocessing
#     process_args = []
#     suffixes = ['', '-gen']
#     gpu_counter = count()
#     for model_name, dataset, dataset_bounds in zip(models, datasets, bounds):
#         for data, _bounds, suffix in zip(dataset, dataset_bounds, suffixes):
#             for bound in _bounds:
#                 process_args.append((
#                     f'../../checkpoints/finale_models/{model_name}',
#                     data,
#                     10,
#                     f'../../data/dpo_dataset/{model_name}{suffix}',
#                     f'{bound[0]},{bound[1]}',
#                     batch_size,
#                     0, #next(gpu_counter) % 8,
#                     True,
#                 ))
#
#     # Use multiprocessing Pool for parallel execution
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         # Map generate_tts function to all prepared arguments
#         pool.starmap(generate_tts, process_args)
#
#
# @click.command()
# @click.option('--num-processes', default=2, help='Number of parallel processes')
# @click.option('--batch-size', default=100, help='Size of parquet file to save')
# def main(num_processes: int, batch_size: int):
#     """
#     Main entry point for parallel TTS dataset generation.
#
#     Args:
#         num_processes (int): Number of parallel processes
#         batch_size (int): Batch size for processing
#     """
#     run_parallel_generation(num_processes, batch_size)
#
#
# if __name__ == "__main__":
#     main()