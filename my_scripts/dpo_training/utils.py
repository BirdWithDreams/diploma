import os
from copy import deepcopy
from itertools import count, product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from TTS.utils.io import load_fsspec

project_root = r'/workspace/Projects/diploma'


def my_formatter(root_path, meta_file, **kwargs):
    meta_file_path = os.path.join(root_path, meta_file)
    meta_df = pd.read_parquet(meta_file_path)
    items = []
    for _, row in meta_df.iterrows():
        text = row['text']
        if not isinstance(text, str):
            continue

        if len(text.strip()) == 0:
            continue

        wav_file = os.path.join(root_path, "wavs", row['audio_id'])
        speaker_name = str(row['speaker_id'])
        items.append(
            {
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path,
                "mel_cond_w": torch.LongTensor(row['mel_cond_w']),
                "mel_cond_l": torch.LongTensor(row['mel_cond_l']),
            }
        )
    return items


epoch_counter = count()
best_cer = float('inf')
best_secs = float('-inf')
best_utmos = float('-inf')


def evaluate_model(trainer, config_dataset, save_dir):
    if next(epoch_counter) % 3 == 0:
        from evaluate_tts import compute_text_metrics, compute_audio_metric, pipe

        global best_cer
        global best_secs
        global best_utmos

        dataset_path = Path(config_dataset.path)
        test_file = config_dataset.meta_file_val
        if test_file.endswith(".csv"):
            test_df = pd.read_csv(dataset_path / test_file)
        elif test_file.endswith(".parquet"):
            test_df = pd.read_parquet(dataset_path / test_file)

        model = trainer.model.xtts
        model.gpt.init_gpt_for_inference(kv_cache=model.args.kv_cache, use_deepspeed=False)
        model.gpt.eval()
        training = model.gpt.training

        metrics = []
        for id_, row in tqdm(test_df.iterrows(), total=len(test_df)):
            prompt = row['text']
            speaker_wav = dataset_path / 'wavs' / (row['audio_id'] + '.wav')
            output = model.synthesize(
                prompt,
                model.config,
                speaker_wav=speaker_wav.as_posix(),
                language='en',
            )

            wave = output['wav']
            wave = np.array(wave, dtype=np.float32)

            transcription = pipe(wave)
            transcription = transcription['text']

            text_metrics = compute_text_metrics(prompt, transcription)
            secs_metric, utmos_metric = compute_audio_metric((wave, 22050), "LJ001-0001.wav", dataset_path)

            row = {
                'prompt_id': id_,
            }
            row |= text_metrics
            row |= {
                'secs': secs_metric,
                'utmos': utmos_metric,
            }

            metrics.append(row)

        metrics = pd.DataFrame(metrics)
        metrics = metrics.groupby('prompt_id')[['cer', 'mer', 'wer', 'wil', 'wip', 'secs', 'utmos']].mean()
        metrics = metrics.mean()
        trainer.dashboard_logger.add_scalars('eval', metrics.to_dict(), trainer.total_steps_done)
        del model.gpt.gpt_inference
        del model.gpt.gpt.wte

        if metrics['cer'] < best_cer:
            best_cer = metrics['cer']
            cer_state_dict = deepcopy(model.cpu().state_dict())
            checkpoint_path = Path(project_root) / 'runs' / save_dir / 'cer'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / 'model.pth').unlink(missing_ok=True)

            torch.save(cer_state_dict, str(checkpoint_path / 'model.pth'))

        if metrics['secs'] > best_secs:
            best_secs = metrics['secs']
            secs_state_dict = deepcopy(model.cpu().state_dict())
            checkpoint_path = Path(project_root) / 'runs' / save_dir / 'secs'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / 'model.pth').unlink(missing_ok=True)

            torch.save(secs_state_dict, str(checkpoint_path / 'model.pth'))

        if metrics['utmos'] > best_utmos:
            best_utmos = metrics['utmos']
            utmos_state_dict = deepcopy(model.cpu().state_dict())
            checkpoint_path = Path(project_root) / 'runs' / save_dir / 'utmos'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / 'model.pth').unlink(missing_ok=True)

            torch.save(utmos_state_dict, str(checkpoint_path / 'model.pth'))

        model.cuda()
        model.gpt.train(training)


def get_compatible_checkpoint_state_dict(model_path):
    checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))['model']
    # remove xtts gpt trainer extra keys
    ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
    for key in list(checkpoint.keys()):
        # check if it is from the coqui Trainer if so convert it
        if key.startswith("xtts."):
            new_key = key.replace("xtts.", "")
            checkpoint[new_key] = checkpoint[key]
            del checkpoint[key]
            key = new_key

        # remove unused keys
        if key.split(".")[0] in ignore_keys:
            del checkpoint[key]

        if 'gpt_inference' in key or 'gpt.wte' in key:
            del checkpoint[key]

    return checkpoint
