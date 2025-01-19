import os
from copy import deepcopy
from itertools import product, count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import wandb
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples, formatters
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

from my_logger import WandbLogger
from dpo_training.evaluate_tts import compute_text_metrics, compute_audio_metric, pipe

project_root = r'/workspace/Projects/diploma'


def my_formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    meta_file_path = os.path.join(root_path, meta_file)
    meta_df = pd.read_csv(meta_file_path)
    items = []
    for _, row in meta_df.iterrows():
        text = row['whisper_transcription']
        if not isinstance(text, str):
            continue

        if len(text.strip()) == 0:
            continue

        audio_id = row['audio_id']
        if not audio_id.endswith('.wav'):
            audio_id = audio_id + '.wav'

        wav_file = os.path.join(root_path, "wavs", audio_id)
        speaker_name = str(row['speaker_id'])
        items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


epoch_counter = count()
best_cer = float('inf')
best_secs = float('-inf')
best_utmos = float('-inf')

base_checkpoints_path = Path(project_root) / 'runs' / 'triple-fn-checkpoints'

def evaluate_model(trainer):
    if next(epoch_counter) % 4 == 0:
        global best_cer
        global best_secs
        global best_utmos

        prompts = pd.read_csv("../data/my_tests/prompts.csv", index_col=0)
        model = trainer.model.xtts
        model.gpt.init_gpt_for_inference(kv_cache=model.args.kv_cache, use_deepspeed=False)
        model.gpt.eval()
        training = model.gpt.training

        metrics = []
        for gen_i, (id_, prompt) in tqdm(product(range(10), enumerate(prompts['prompt'])), total=len(prompts) * 10):
            output = model.synthesize(
                prompt,
                model.config,
                speaker_wav="../data/dpo_dataset/wavs/LJ001-0001.wav",
                language='en',
            )

            wave = output['wav']
            wave = np.array(wave, dtype=np.float32)

            transcription = pipe(wave)
            transcription = transcription['text']

            text_metrics = compute_text_metrics(prompt, transcription)
            secs_metric, utmos_metric = compute_audio_metric((wave, 22050), "LJ001-0001.wav", lg_config_dataset.path)

            row = {
                'prompt_id': id_,
                'gen': gen_i,
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
            checkpoint_path = base_checkpoints_path / 'cer'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / 'model.pth').unlink(missing_ok=True)

            torch.save(cer_state_dict, str(checkpoint_path / 'model.pth'))

        if metrics['secs'] > best_secs:
            best_secs = metrics['secs']
            secs_state_dict = deepcopy(model.cpu().state_dict())
            checkpoint_path = base_checkpoints_path / 'secs'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / 'model.pth').unlink(missing_ok=True)

            torch.save(secs_state_dict, str(checkpoint_path / 'model.pth'))

        if metrics['utmos'] > best_utmos:
            best_utmos = metrics['utmos']
            utmos_state_dict = deepcopy(model.cpu().state_dict())
            checkpoint_path = base_checkpoints_path / 'utmos'
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / 'model.pth').unlink(missing_ok=True)

            torch.save(utmos_state_dict, str(checkpoint_path / 'model.pth'))

        model.cuda()
        model.gpt.train(training)


# Logging parameters
RUN_NAME = "LG_ASR _FT"
PROJECT_NAME = "xTTS-training"
DASHBOARD_LOGGER = "wandb"
LOGGER_URI = None


# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(project_root, "runs/training")

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = False  # if True it will star with evaluation
BATCH_SIZE = 4  # set here the batch size
GRAD_ACUMM_STEPS = 252 // BATCH_SIZE + 1  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
voxpopuli_config_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="facebook-voxpopuli",
    path=os.path.join(project_root, "data/facebook_voxpopuli"),  # Updated to use the Hugging Face dataset
    meta_file_train=r"train_metadata.csv",
    meta_file_val='test_metadata.csv',
    language="en",
)
lg_config_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="keithito_lj_speech",
    path=os.path.join(project_root, "data/keithito_lj_speech"),  # Updated to use the Hugging Face dataset
    meta_file_train=r"train_metadata.csv",
    meta_file_val='test_metadata.csv',
    language="en",
)
vctk_config_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="VCTK-Corpus",
    path=os.path.join(project_root, "data/VCTK-Corpus"),  # Updated to use the Hugging Face dataset
    meta_file_train=r"train_metadata.csv",
    meta_file_val='test_metadata.csv',
    language="en",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [
    # voxpopuli_config_dataset,
    lg_config_dataset,
    # vctk_config_dataset,
]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# Training sentences generations
SPEAKER_REFERENCE = [os.path.join(project_root, "data/speakers/LJ001-0001.wav")]
# SPEAKER_REFERENCE = list(Path('../data/speakers/').glob('*.wav'))
LANGUAGE = lg_config_dataset.language


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training on facebook/Voxpopuli-Dataset
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=1000,
        eval_split_size=0.1,
        print_step=100,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
        wandb_entity='kpi-msai',
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=my_formatter,
    )

    my_logger = WandbLogger(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config=config,
        entity=config.wandb_entity,
        )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            # continue_path='./run/training/GPT_XTTS_v2.0_Voxpopuli_FT-October-19-2024_01+38PM-c7202bff',
            # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        dashboard_logger=my_logger,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        # callbacks={'on_epoch_end': evaluate_model}
    )
    trainer.fit()


if __name__ == "__main__":
    main()
