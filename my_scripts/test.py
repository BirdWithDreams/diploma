# import torch
# from tqdm import tqdm
# from TTS.api import TTS
#
# # Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
#
# # List available 🐸TTS models
# print(TTS().list_models())
#
# # Init TTS
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
#
# # Run TTS
# # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# # Text to speech list of amplitude values as output
# # wav = tts.tts(text="Hello world!", speaker_wav="sample.m4a", language="en")
# # Text to speech to a file
# with open('../data/my_tests/prompts', 'r') as f:
#     prompts = f.readlines()
#
# for prompt in tqdm(prompts[50:]):
#     number = prompt.split(' ', maxsplit=1)[0]
#     prompt = prompt.split(' ', maxsplit=1)[1]
#
#     tts.tts_to_file(
#         text=prompt,
#         speaker_wav=r"D:\Учеба\КПИ\Diploma\TTS\tests\data\ljspeech\wavs\LJ001-0003.wav",
#         language="en",
#         file_path=f"my_tests/audio/{number}.wav"
#     )


# import pandas as pd
#
# data = pd.read_csv('../data/dpo_dataset/llama.csv', index_col=0)
# data = data.sample(frac=1, random_state=42)
# data.to_csv('../data/dpo_dataset/llama_shuffled.csv')


# import torch
# from TTS.api import TTS
# from TTS.utils.io import load_fsspec
#
#
# def get_compatible_checkpoint_state_dict(model_path):
#     checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))
#     # remove xtts gpt trainer extra keys
#     ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
#     for key in list(checkpoint.keys()):
#         # check if it is from the coqui Trainer if so convert it
#         if key.startswith("xtts."):
#             new_key = key.replace("xtts.", "")
#             checkpoint[new_key] = checkpoint[key]
#             del checkpoint[key]
#             key = new_key
#
#         # remove unused keys
#         if key.split(".")[0] in ignore_keys:
#             del checkpoint[key]
#
#     return checkpoint
#
#
# # model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
# # model = TTS(
# #     model_path='/workspace/Projects/diploma/runs/dpo-checkpoints/test/',
# #     config_path='/workspace/Projects/diploma/runs/training/GPT_XTTS_Triple_Dataset_FT-January-07-2025_02+42PM-3c890633/config.json'
# # )
#
# state_dict_1 = torch.load('/workspace/Projects/diploma/checkpoints/finale_models/vctk-asr/model.pth')
# state_dict_1 = {'model':state_dict_1}
# # torch.save(state_dict_1, '/workspace/Projects/diploma/checkpoints/finale_models/lg-asr/model.pth')
#
# state_dict_2 = torch.load('/workspace/Projects/diploma/checkpoints/finale_models/lg-asr/model.pth', map_location=torch.device('cpu'))
# state_dict_2 = {'model':state_dict_2}
#
# pass
#
# # from itertools import product
# #
# # import torch
# # from pathlib import Path
# #
# # import tqdm
# #
# # root_path = Path('/workspace/Projects/diploma')
# # dirs = ['dpo-checkpoints',  'triple-fn-checkpoints']
# # metrics = ['cer', 'secs', 'utmos']
# #
# # checkpoints_path = root_path / 'checkpoints'
# # runs_path = root_path / 'runs'
# #
# #
# # for d, m in tqdm.tqdm(product(dirs, metrics)):
# #     s = torch.load(runs_path / d/m/'model.pth')
# #     s = {'model': s}
# #     output_path = checkpoints_path / d / m
# #     output_path.mkdir(parents=True, exist_ok=True)
# #     torch.save(s, output_path / 'model.pth')
#
# # import os
# # from pathlib import Path
# #
# # path = Path('/workspace/Projects/diploma/data/VCTK-Corpus')
# # for file in path.glob('*checkpoints*-metrics.csv'):
# #     # Create the new filename with 'vp-' prefix
# #     new_name = 'vctk-' + file.name
# #     # Create new Path object with the parent directory and new filename
# #     new_path = file.parent / new_name
# #     # Rename the file
# #     os.rename(file, new_path)
#
#
# from pathlib import Path
# import pandas as pd
#
#
# path_vctk = Path('/workspace/Projects/diploma/data/VCTK-Corpus')
#
# df = pd.read_csv(path_vctk / 'metadata.csv')
# df = df.iloc[:, 1:]
# train = pd.read_csv(path_vctk / 'train_metadata.csv')
# test = pd.read_csv(path_vctk / 'test_metadata.csv')
# speakers = pd.read_csv(path_vctk / 'speakers.csv')
#
# train = df[df['audio_id'].isin(train['audio_id'])]
# test = df[df['audio_id'].isin(test['audio_id'])]
# speakers = df[df['audio_id'].isin(speakers['audio_id'])]
#
# train.to_csv(path_vctk / 'train_metadata.csv', index=False)
# test.to_csv(path_vctk / 'test_metadata.csv', index=False)
# speakers.to_csv(path_vctk / 'speakers.csv', index=False)
# df.to_csv(path_vctk / 'metadata.csv', index=False)


from pathlib import Path
import torch

fn_path = Path('/workspace/Projects/diploma/runs/training')

models = [
    'DPO_VCTK_ASR_Augmented_Training-February-08-2025_07+09PM-4268688b/best_model.pth',
    'DPO_VCTK_ASR_Augmented_Training-February-08-2025_07+09PM-4268688b/checkpoint_461754.pth',
]

out_path = Path('/workspace/Projects/diploma/checkpoints/finale_models')

for name, model in zip(['asr-vctk-dpo-augmented-best', 'asr-vctk-dpo-augmented-last'], models):
    path = fn_path / model
    state_dict = torch.load(path, map_location='cpu')
    state_dict = {'model': {k:v for k, v in state_dict['model'].items() if not k.startswith('ref_xtts')}}
    out = out_path / name / 'model.pth'
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(out))


# import pandas as pd
# from pathlib import Path
# import shutil
# import itertools
# import random
#
# def distribute_prompts(num_users, prompts, prompts_per_user):
#     """
#     Distributes prompts among users ensuring each prompt is used and distribution is approximately even.
#
#     Args:
#         num_users (int): Number of users
#         prompts (list): List of prompt identifiers/texts
#         prompts_per_user (int): Number of prompts to assign to each user
#
#     Returns:
#         dict: Mapping of user IDs to lists of assigned prompts
#     """
#     # Calculate total number of prompt assignments needed
#     total_assignments = num_users * prompts_per_user
#
#     # Calculate how many times each prompt should be used ideally
#     base_repetitions = total_assignments // len(prompts)
#     extra_assignments = total_assignments % len(prompts)
#
#     # Create a pool of prompts with proper repetitions
#     prompt_pool = []
#     for i, prompt in enumerate(prompts):
#         # Add base repetitions for each prompt
#         repetitions = base_repetitions
#         # Distribute extra assignments evenly
#         if i < extra_assignments:
#             repetitions += 1
#         prompt_pool.extend([prompt] * repetitions)
#
#     # Shuffle the pool
#     random.shuffle(prompt_pool)
#
#     # Distribute prompts to users
#     user_assignments = {}
#     for user_id in range(num_users):
#         start_idx = user_id * prompts_per_user
#         end_idx = start_idx + prompts_per_user
#         user_assignments[user_id] = prompt_pool[start_idx:end_idx]
#
#     return user_assignments
#
#
# data = pd.read_csv('/workspace/Projects/diploma/data/VCTK-Corpus/metadata.csv')
# gen = pd.read_csv('/workspace/Projects/diploma/data/dpo_dataset/llama_shuffled.csv', index_col=0)
#
# speakers = data.groupby('speaker_id').head(1)
# in_path = Path('/workspace/Projects/diploma/data/VCTK-Corpus/wavs')
# out_path = Path('/workspace/Projects/diploma/data/VCTK-Corpus_gen/wavs')
#
#
# for speaker_audio in speakers['audio_id']:
#     shutil.copyfile(in_path / speaker_audio, out_path / speaker_audio)
#
#
# assignments = distribute_prompts(len(speakers), list(gen['llama3.2-3b-prompt'].items()), 506)
#
# metadata = []
# for speaker, prompt_pair in assignments.items():
#     for prompt_id, prompt in set(prompt_pair):
#         metadata.append({
#             'speaker_id': speaker,
#             'text': prompt.strip(),
#             'prompt_id': prompt_id,
#         })
#
# metadata = pd.DataFrame(metadata)
# metadata['speaker_id'] = metadata['speaker_id'].apply(lambda x: speakers.iloc[x]['speaker_id'])
# metadata = metadata.merge(speakers[['speaker_id', 'audio_id']], on='speaker_id', how='left')
# metadata.to_csv(out_path.parent / 'metadata.csv', index=False)


# import pandas as pd
# from pathlib import Path
# import shutil
# import itertools
# import random
#
# data = pd.read_csv('/workspace/Projects/diploma/data/keithito_lj_speech/metadata.csv')
# gen = pd.read_csv('/workspace/Projects/diploma/data/dpo_dataset/llama_shuffled.csv', index_col=0)
#
# speakers = data.groupby('speaker_id').head(1)
# in_path = Path('/workspace/Projects/diploma/data/keithito_lj_speech/wavs')
# out_path = Path('/workspace/Projects/diploma/data/keithito_lj_speech_gen/wavs')
#
# metadata = []
# for audio_id, (prompt_id, prompt) in zip(itertools.cycle(['LJ001-0001.wav', 'LJ001-0002.wav', 'LJ001-0003.wav']), gen['llama3.2-3b-prompt'].items()):
#     metadata.append({
#         'audio_id': audio_id,
#         'speaker_id': 'lg_speaker',
#         'text': prompt.strip(),
#         'prompt_id': prompt_id,
#     })
#
# metadata = pd.DataFrame(metadata)
# metadata.to_csv(out_path.parent / 'metadata.csv', index=False)


# from pathlib import Path
# import pandas as pd
#
#
# def filter_df(df):
#     prompt_count = df.groupby(['speaker_id', 'prompt_id']).count()['gen_id']
#     prompt_count = prompt_count[prompt_count >= 10]
#     prompt_count = prompt_count.reset_index()['prompt_id'].unique()
#     return df[df['prompt_id'].isin(prompt_count)]
#
# path = Path(r'../data/dpo_dataset/lg-asr-gen')
# lg_asr_gen = pd.concat([pd.read_parquet(file) for file in list(path.glob('*'))], axis=0)
#
# path = Path(r'../data/dpo_dataset/lg-asr')
# lg_asr = pd.concat([pd.read_parquet(file) for file in list(path.glob('*'))], axis=0)
#
#
# path = Path(r'../data/dpo_dataset/vctk-asr')
# vctk_asr = pd.concat([pd.read_parquet(file) for file in list(path.glob('*'))], axis=0)
#
# path = Path(r'../data/dpo_dataset/vctk-asr-gen')
# vctk_asr_gen = pd.concat([pd.read_parquet(file) for file in list(path.glob('*'))], axis=0)
#
# lg_asr = filter_df(lg_asr)
# vctk_asr = filter_df(vctk_asr)
#
# lg_asr_gen = filter_df(lg_asr_gen)
# vctk_asr_gen = filter_df(vctk_asr_gen)
#
# lg_asr_gen.to_parquet('../data/dpo_dataset/lg_asr_gen.parquet', index=False)
# lg_asr.to_parquet('../data/dpo_dataset/lg_asr.parquet', index=False)
#
# vctk_asr_gen.to_parquet('../data/dpo_dataset/vctk_asr_gen.parquet', index=False)
# vctk_asr.to_parquet('../data/dpo_dataset/vctk_asr.parquet', index=False)
