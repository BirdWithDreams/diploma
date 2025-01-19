import os
import random
from itertools import product
from pathlib import Path

import fire
import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import tqdm
import transformers
from huggingface_hub import hf_hub_download
from loguru import logger
from pydub import AudioSegment
from transformers import pipeline

from TTS.api import TTS

# Configure loguru logger
logger.add("../logs/evaluate_tts_{time}.log", rotation="10 MB", level="INFO")


def set_seed(random_seed=1234):
    logger.info(f"Setting random seed to {random_seed}")
    # set deterministic inference
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    transformers.set_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)
    logger.debug("Random seed set successfully")


set_seed()

CUDA_AVAILABLE = torch.cuda.is_available()
device = "cuda" if CUDA_AVAILABLE else "cpu"
logger.info(f"Using device: {device}")

logger.info("Loading ECAPA2 model")
ecapa2_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
ecapa2 = torch.jit.load(ecapa2_file, map_location='cpu').to(device)
logger.debug("ECAPA2 model loaded successfully")

logger.info("Loading UTMOS model")
mos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
logger.debug("UTMOS model loaded successfully")

model_name = "openai/whisper-medium"
logger.info(f"Setting up ASR pipeline with model: {model_name}")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    chunk_length_s=30,
    device='cuda',
)
logger.debug("ASR pipeline set up successfully")

metric_funcs = {
    'cer': jiwer.cer,
    'mer': jiwer.mer,
    'wer': jiwer.wer,
    'wil': jiwer.wil,
    'wip': jiwer.wip,
}

transforms = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def torch_rms_norm(wav, db_level=-27.0):
    r = 10 ** (db_level / 20)
    a = torch.sqrt((wav.size(-1) * (r ** 2)) / torch.sum(wav ** 2))
    return wav * a


def get_ecapa2_spk_embedding(path=None, audio=None, ref_dBFS=None, model_sr=16000):
    if path is not None and path.exists():
        audio, sr = torchaudio.load(path)
        if audio.size(1) == 0:
            return None
    elif audio is not None:
        audio, sr = audio
        audio = torch.FloatTensor(audio).unsqueeze(0)
    else:
        logger.error("Neither path nor audio provided for ECAPA2 embedding")
        raise ValueError('One of `path` or `audio` arguments should not be None')

    # sample rate of 16 kHz expected
    if sr != model_sr:
        logger.debug(f"Resampling audio from {sr} Hz to {model_sr} Hz")
        audio = torchaudio.functional.resample(audio, sr, model_sr)

    # RMS norm based on the reference audio dBFS it make all models output in the same db level and it avoid issues
    if ref_dBFS is not None:
        audio = torch_rms_norm(audio, db_level=ref_dBFS)

    # compute speaker embedding
    embed = ecapa2(audio.to(device))
    # ensures that l2 norm is applied on output
    embed = torch.nn.functional.normalize(embed, p=2, dim=1)
    return embed.cpu().detach().squeeze().numpy()


def compute_UTMOS(path=None, audio=None, ref_dBFS=None):
    logger.debug("Computing UTMOS score")
    if path is not None:
        audio, sr = librosa.load(path, sr=None, mono=True)
    elif audio is not None:
        audio, sr = audio
    else:
        logger.error("Neither path nor audio provided for UTMOS computation")
        raise ValueError('One of `path` or `audio` arguments should not be None')

    audio = torch.from_numpy(audio).unsqueeze(0)
    # RMS norm based on the reference audio dBFS it make all models output in the same db level and it avoid issues
    audio = torch_rms_norm(audio, db_level=ref_dBFS)
    # predict UTMOS
    score = mos_predictor(audio.to(device), sr).item()
    logger.debug(f"UTMOS score computed: {score}")
    return score


def compute_ref_secs(root_path, speakers=None):
    logger.info(f"Computing reference speaker embeddings for {root_path}")
    root_path = Path(root_path)
    test_file = root_path / 'test_metadata.csv'
    test_meta_df = pd.read_csv(test_file)
    if speakers is not None and isinstance(speakers, list):
        test_meta_df = test_meta_df[test_meta_df['speaker_id'].isin(speakers)]
    test_sample = test_meta_df.groupby('speaker_id').sample(15, replace=True)
    test_sample = test_sample.groupby('speaker_id')['audio_id'].agg(list)

    ref_scores = {}
    for speaker_id, speaker_audios in test_sample.items():
        speaker_embs = []

        ref_audio_id = speaker_audios[0]
        if ref_audio_id.endswith('.wav'):
            audio_path = root_path / 'wavs' / ref_audio_id
        else:
            audio_path = root_path / 'wavs' / (ref_audio_id + '.wav')

        ref_dBFS = AudioSegment.from_file(audio_path).dBFS

        for speaker_audio in speaker_audios:
            ref_audio_id = speaker_audio
            if ref_audio_id.endswith('.wav'):
                audio_path = root_path / 'wavs' / ref_audio_id
            else:
                audio_path = root_path / 'wavs' / (ref_audio_id + '.wav')
            emb = get_ecapa2_spk_embedding(audio_path, ref_dBFS=ref_dBFS)
            if emb is not None:
                speaker_embs.append(emb.reshape((1, -1)))

        speaker_embs = np.concatenate(speaker_embs)
        speakre_secs = speaker_embs @ speaker_embs.T
        speakre_secs = np.triu(speakre_secs, 1)
        ref_scores[speaker_id] = speakre_secs[speakre_secs != 0].mean()

    logger.info("Reference speaker embeddings computed successfully")
    return ref_scores


def compute_audio_metric(gen_wave, ref_audio_id, root_path):
    logger.debug(f"Computing audio metrics for reference audio: {ref_audio_id}")

    if ref_audio_id.endswith('.wav'):
        audio_path = root_path / 'wavs' / ref_audio_id
    else:
        audio_path = root_path / 'wavs' / (ref_audio_id + '.wav')

    ref_dBFS = AudioSegment.from_file(audio_path.resolve()).dBFS
    gen_emb = get_ecapa2_spk_embedding(audio=gen_wave, ref_dBFS=ref_dBFS)
    ref_emb = get_ecapa2_spk_embedding(path=audio_path, ref_dBFS=ref_dBFS)
    secs = gen_emb @ ref_emb

    utmos = compute_UTMOS(audio=gen_wave, ref_dBFS=ref_dBFS)
    logger.debug(f"Audio metrics computed: SECS={secs}, UTMOS={utmos}")
    return secs, utmos


def compute_text_metrics(text, transcription):
    logger.debug("Computing text metrics")
    m = {}
    for metric_name, metric_func in metric_funcs.items():
        m[metric_name] = metric_func(
            text,
            transcription,
            truth_transform=transforms,
            hypothesis_transform=transforms,
        )
    logger.debug(f"Text metrics computed: {m}")
    return m


def eval_tts(
        model_name,
        prompt_csv='../data/my_tests/prompts.csv',
        sample_size=10,
        dataset_path='../data/keithito_lj_speech',
        model_path='../checkpoints/good_dataset/',
        output_name='metrics'
):
    logger.info(f"Starting TTS evaluation for model: {model_name}")
    model_path = Path(model_path).resolve()
    prompts = pd.read_csv(prompt_csv, index_col=0)
    logger.info(f"Loaded {len(prompts)} prompts from {prompt_csv}")

    logger.info(f"Loading TTS model from {model_path}")

    if model_name =='base_xtts_v2':
        logger.info('Use base XTTS model')
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
    else:
        logger.info('Use model from checkpoint')
        model = TTS(
            # model_name='xtts_v2',
            model_path=model_path.as_posix(),
            config_path='/workspace/Projects/diploma/runs/training/GPT_XTTS_Triple_Dataset_FT-January-07-2025_02+42PM-3c890633/config.json',#(model_path / 'config.json').as_posix(),
        ).to('cuda')
    logger.debug("TTS model loaded successfully")

    dataset_path = Path(dataset_path).resolve()
    speakers_path = dataset_path / 'speakers.csv'
    speakers = pd.read_csv(speakers_path)
    logger.info(f"Loaded {len(speakers)} speakers from {speakers_path}")

    logger.info("Computing reference speaker embeddings")
    ref_speakers_secs = compute_ref_secs(dataset_path, speakers['speaker_id'].tolist())

    metrics = []
    total_iterations = len(prompts) * len(speakers) * sample_size
    logger.info(f"Starting evaluation loop. Total iterations: {total_iterations}")
    for (id_, prompt), (_, speaker), gen_i in tqdm.tqdm(
            product(
                prompts.iterrows(),
                speakers.iterrows(),
                range(sample_size)
            ),
            total=total_iterations
    ):
        speaker_name = speaker['speaker_id']
        speaker_wav = dataset_path / 'wavs' / (speaker['audio_id'] + '.wav')

        logger.debug(f"Generating TTS for prompt {id_}, speaker {speaker_name}, iteration {gen_i}")
        wave, gpt_codes = model.tts(
            text=prompt['prompt'],
            speaker_wav=speaker_wav.as_posix(),
            language="en",
        )
        try:
            wave = np.array(wave, dtype=np.float32)
        except ValueError:
            wave = sum(wave, start=[])
            wave = np.array(wave, dtype=np.float32)

        logger.debug("Transcribing generated audio")
        transcription = pipe(wave)
        transcription = transcription['text']

        logger.debug("Computing metrics")
        text_metrics = compute_text_metrics(prompt['prompt'], transcription)
        secs_metric, utmos_metric = compute_audio_metric((wave, 22050), speaker['audio_id'], dataset_path)

        row = {
            'prompt_id': id_,
            'speaker_id': speaker_name,
            'gen': gen_i,
            'model_name': model_name
        }
        row |= text_metrics
        row |= {
            'ref_secs': ref_speakers_secs[speaker_name],
            'secs': secs_metric,
            'utmos': utmos_metric,
        }

        row |= {
            'prompt': prompt['prompt'],
            'transcription': transcription,
        }

        metrics.append(row)
        logger.debug(f"Metrics for current iteration: {row}")

    metrics_df = pd.DataFrame(metrics)
    output_path = dataset_path / f'{output_name}.csv'
    logger.info(f"Saving metrics to {output_path}")
    metrics_df.to_csv(output_path, index=False)
    logger.info("Evaluation completed successfully")


if __name__ == '__main__':
    logger.info("Script started")
    fire.Fire(eval_tts)
    logger.info("Script finished")
