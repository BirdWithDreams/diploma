import os
import sys
import traceback
from itertools import product
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tqdm
from loguru import logger

# Configure logging with more detailed settings
logger.remove()  # Remove default logger
logger.add(
    "../../logs/generate_dpo_data.log",
    rotation="10 MB",
    level="INFO",
)
logger.add(sys.stderr, level="INFO")  # Also log to console

project_dir = Path(__file__).resolve().parents[2]
default_prompt_path = project_dir / 'data' / 'VCTK-Corpus'
default_model_path = project_dir / 'checkpoints' / 'finale_models' / 'vctk-asr'
default_output_path = project_dir / 'data' / 'dpo_dataset' / 'vctk-asr'


def generate_tts(model_path, dataset_path, sample_size, output_folder, prompts_bounds, batch_size, gpu, extra):
    """
    Generate TTS samples with comprehensive logging and error handling.

    Args:
        model_path (str): Path to the TTS model
        dataset_path (str): Path to the input dataset
        sample_size (int): Number of samples to generate per input
        output_folder (str): Folder to save generated samples
        prompts_bounds (str): Range of prompts to process
        batch_size (int): Number of samples to save in each batch
        gpu (int): GPU device to use
        extra (bool): Flag to check previous generation info
    """
    # Log script start with configuration details
    logger.info("Starting TTS Data Generation Script")
    logger.info(f"Configuration:")
    logger.info(f"  Model Path: {model_path}")
    logger.info(f"  Dataset Path: {dataset_path}")
    logger.info(f"  Sample Size: {sample_size}")
    logger.info(f"  Output Folder: {output_folder}")
    logger.info(f"  Prompts Bounds: {prompts_bounds}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  GPU Device: {gpu}")
    logger.info(f"  Extra Mode: {extra}")

    # Set GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    logger.info(f"Setting CUDA_VISIBLE_DEVICES to {gpu}")

    # Import TTS and evaluation modules
    from TTS.api import TTS
    from evaluate_tts import compute_text_metrics, compute_audio_metric, pipe

    # Resolve paths
    model_path = Path(model_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    logger.info(f"Resolved model path: {model_path}")
    logger.info(f"Resolved dataset path: {dataset_path}")

    # Load and slice metadata
    try:
        metadata = pd.read_csv(dataset_path / 'metadata.csv')
        logger.info(f"Loaded metadata: {len(metadata)} total entries")

        # Process prompts bounds
        low, high = prompts_bounds.split(',')
        low = 0 if low == 'na' else int(low)
        high = len(metadata) if high == 'na' else int(high)

        logger.info(f"Processing entries from index {low} to {high}")
        metadata = metadata.iloc[low:high]
    except Exception as e:
        logger.error(f"Error processing metadata: {e}")
        logger.error(traceback.format_exc())
        raise

    # Load TTS model
    try:
        logger.info("Loading TTS model")
        # model = TTS(
        #     model_path=model_path.as_posix(),
        #     config_path=(model_path / 'config.json').as_posix(),
        # ).to(f'cuda')
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
        logger.info("TTS model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        logger.error(traceback.format_exc())
        raise

    # Prepare output path
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path created: {output_path}")

    # Handle extra mode context
    context = None
    if extra:
        try:
            name = output_path.stem
            context = pd.read_parquet(output_path.parent / f'{name.replace("-", "_")}.parquet')[['audio_id', 'prompt_id']]
            context = list(context.drop_duplicates().itertuples(index=False, name=None))
            logger.info(f"Loaded {len(context)} existing prompt IDs in extra mode")
        except Exception as e:
            logger.warning(f"Error in extra mode processing: {e}")
            context = None

    dpo_samples = []
    total_processed = 0
    total_errors = 0

    # Main generation loop with comprehensive logging
    for i, ((id_, row), gen_i) in tqdm.tqdm(
            enumerate(product(
                metadata.iterrows(),
                range(sample_size)
            )),
            total=len(metadata) * sample_size
    ):
        try:
            # Extract speaker and audio details
            speaker_name = row['speaker_id']
            audio_id = row['audio_id'] if row['audio_id'].endswith('.wav') else row['audio_id'] + '.wav'
            speaker = dataset_path / 'wavs' / audio_id

            # Skip already processed entries in extra mode
            if extra and context is not None and (speaker.name, id_) in context:
                logger.debug(f"Skipping already processed prompt ID: {id_}")
                continue

            # Get prompt text
            try:
                prompt = row['whisper_transcription']
            except KeyError:
                prompt = row['text']

            logger.debug(f"Processing audio: {speaker.name}, Prompt: {prompt}")

            # Generate TTS
            wav, gpt_codes = model.tts(
                text=prompt,
                speaker_wav=speaker,
                language="en",
            )
            wav = np.array(wav, dtype=np.float32)
            gpt_codes = np.hstack(gpt_codes)

            # Compute metrics
            secs_metric, utmos_metric = compute_audio_metric((wav, 22050), speaker.name, dataset_path)
            transcription = pipe(wav, generate_kwargs={"language": "english", 'return_timestamps': True})['text']
            text_metrics = compute_text_metrics(transcription, prompt)

            # Prepare sample data
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
                } | text_metrics
            )

            total_processed += 1

        except Exception as e:
            total_errors += 1
            logger.error(f'Exception processing {speaker.name} with prompt "{prompt}": {str(e)}')
            logger.error(traceback.format_exc())

        # Batch saving mechanism
        if (i + 1) % batch_size == 0:
            try:
                dpo_samples_df = pd.DataFrame(dpo_samples)
                batch_file = output_path / f'batch_{low + i}.parquet'
                dpo_samples_df.to_parquet(batch_file, engine='pyarrow')

                logger.info(f"Saved batch: {batch_file}")
                logger.info(f"Batch stats - Total samples: {len(dpo_samples_df)}")
            except Exception as e:
                logger.error(f"Error saving batch {i}: {str(e)}")
                logger.error(traceback.format_exc())
            finally:
                dpo_samples = []

    # Final batch save
    if dpo_samples:
        try:
            dpo_samples_df = pd.DataFrame(dpo_samples)
            final_batch_file = output_path / f'batch_{low + i}.parquet'
            dpo_samples_df.to_parquet(final_batch_file, engine='pyarrow')

            logger.info(f"Saved final batch: {final_batch_file}")
            logger.info(f"Final batch stats - Total samples: {len(dpo_samples_df)}")
        except Exception as e:
            logger.error(f"Error saving final batch: {str(e)}")
            logger.error(traceback.format_exc())

    # Script completion logging
    logger.info("TTS Data Generation Script Completed")
    logger.info(f"Total Processed: {total_processed}")
    logger.info(f"Total Errors: {total_errors}")
    if total_processed + total_errors > 0:
        logger.info(f"Success Rate: {(total_processed / (total_processed + total_errors)) * 100:.2f}%")
    else:
        logger.info(f"Success Rate: {1 * 100:.2f}%")


@click.command()
@click.option('--model-path', default=default_model_path, help='Path to dataset')
@click.option('--dataset-path', default=default_prompt_path, help='Path to dataset')
@click.option('--sample-size', default=10, help='Number of samples to generate')
@click.option('--output-folder', default=default_output_path, help='Name of output folder')
@click.option('--prompts-bounds', default='na,na', help='Name of output folder')
@click.option('--batch-size', default=100, help='The size of batch to save')
@click.option('--gpu', default=0, help='The number of GPU to use')
@click.option('--extra', default=False, help='Flag to check info about previous generation', is_flag=True)
def main(model_path, dataset_path, sample_size, output_folder, prompts_bounds, batch_size, gpu, extra):
    generate_tts(model_path, dataset_path, sample_size, output_folder, prompts_bounds, batch_size, gpu, extra)


if __name__ == '__main__':
    main()
