import os
from functools import partial

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from dpo_trainer import DPOArgs
from dpo_trainer import DPOTrainer
from my_logger import WandbLogger
from utils import my_formatter, get_compatible_checkpoint_state_dict, evaluate_model

# Logging parameters
RUN_NAME = "DPO_VCTK_ASR_Augmented_Training"
PROJECT_NAME = "xTTS-training"
DASHBOARD_LOGGER = "wandb"
LOGGER_URI = None

project_root = r'/workspace/Projects/diploma'

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(project_root, "runs/training")

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 2  # set here the batch size
GRAD_ACUMM_STEPS = 252 // BATCH_SIZE + 1  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

# Define here the dataset that you want to use for the fine-tuning on.
vctk_dpo_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="vctk_dpo_dataset",
    path=os.path.join(project_root, 'data/VCTK-Corpus'),  # Updated to use the Hugging Face dataset
    meta_file_train=r"dpo_data_train.parquet",
    meta_file_val='dpo_data_test.parquet',
    language="en",
)

vctk_dpo_gen_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="vctk_dpo_gen_dataset",
    path=os.path.join(project_root, 'data/VCTK-Corpus_gen'),  # Updated to use the Hugging Face dataset
    meta_file_train=r"dpo_data.parquet",
    language="en",
)

lg_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="lg_dpo_dataset",
    path=os.path.join(project_root, 'data/keithito_lj_speech'),  # Updated to use the Hugging Face dataset
    meta_file_train=r"train_metadata.csv",
    meta_file_val='test_metadata.csv',
    language="en",
)

lg_dpo_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="lg_dpo_dataset",
    path=os.path.join(project_root, 'data/keithito_lj_speech'),  # Updated to use the Hugging Face dataset
    meta_file_train=r"dpo_data_train.parquet",
    meta_file_val='dpo_data_test.parquet',
    language="en",
)

lg_dpo_gen_dataset = BaseDatasetConfig(
    formatter="my_formatter",
    dataset_name="lg_dpo_dataset",
    path=os.path.join(project_root, 'data/keithito_lj_speech_gen'),  # Updated to use the Hugging Face dataset
    meta_file_train=r"dpo_data.parquet",
    language="en",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [
    vctk_dpo_dataset,
    vctk_dpo_gen_dataset,
    # lg_dpo_dataset,
    # lg_dpo_gen_dataset,
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
SPEAKER_REFERENCE = [
    os.path.join(project_root, "data/speakers/p236_002.wav"),
    # os.path.join(project_root, "data/speakers/LJ001-0001.wav"),

]
# SPEAKER_REFERENCE = list(Path('../data/speakers/').glob('*.wav'))
LANGUAGE = 'en'


def main():
    # init args and config
    model_args = DPOArgs(
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
        beta=0.01,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        epochs=25,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            AR Finetuned GPT XTTS DPO training on VCTK F3 scored dataset with generated texts
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
        lr=1e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [6016 * 4, 6016 * 8, 6016 * 12], "gamma": 0.9, "last_epoch": -1},
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
    model = DPOTrainer.init_from_config(config)
    model.xtts.load_state_dict(get_compatible_checkpoint_state_dict('/workspace/Projects/diploma/checkpoints/finale_models/vctk-asr/model.pth'))
    model.ref_xtts.load_state_dict(get_compatible_checkpoint_state_dict('/workspace/Projects/diploma/checkpoints/finale_models/vctk-asr/model.pth'))

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
        # id='r8nem03s',
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            # restore_path='/workspace/Projects/diploma/runs/training/DPO_LG_Training-December-26-2024_10+12AM-3c890633/',
            # continue_path='/workspace/Projects/diploma/runs/training/DPO_LG_Training-December-26-2024_10+12AM-3c890633',  # './run/training/GPT_XTTS_v2.0_Voxpopuli_FT-November-16-2024_10+27AM-0000000',
            # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=False,  # START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        dashboard_logger=my_logger,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        # callbacks={'on_epoch_end': partial(evaluate_model, config_dataset=lg_dataset, save_dir='lg-base-dpo')}
    )
    trainer.fit()


if __name__ == "__main__":
    main()
