import subprocess
import sys
import shutil
from itertools import product
from pathlib import Path

import pandas as pd
import tqdm

models = {
    # 'best-lg-asr-fn': '../checkpoints/lg_asr_best',
    # 'last-lg-asr-fn': '../checkpoints/lg_asr_last',
    # 'best-vp-fn': '../checkpoints/vox_best',
    # 'last-vp-fn': '../checkpoints/vox_last',
    'base_xtts_v2': 'base_xtts_v2',
    'last-vctk-fn': '../checkpoints/vctk_last',
    'best-vctk-fn': '../checkpoints/vctk_best',
}

datasets = [
    # '../data/facebook_voxpopuli',
    # '../data/keithito_lj_speech',
    '../data/VCTK-Corpus',
]

for model_name, model_path in models.items():
    for dataset in  datasets:
        print(f'start {model_name} model')
        status = subprocess.run([
            'python', 'evaluate_tts.py',
            '--dataset-path', dataset,
            '--model-name', model_name,
            '--model-path', model_path,
            '--output-name', f'{model_name.replace("-fn", "")}-vp-metrics-3'
        ], stdout=sys.stdout)

        if status.returncode == 0:
            print(f'end {model_name} model')


# %%
# import pandas as pd
#
# dataset_path = Path('../data/keithito_lj_speech')
# metadata = pd.read_csv(dataset_path / 'metadata.csv', index_col=0)
#
# train_metadata = pd.read_csv(dataset_path / 'train_metadata.csv')
# test_metadata = pd.read_csv(dataset_path / 'test_metadata.csv')
#
# wh_metadata = pd.read_csv('../data/metadata.csv')
# union = metadata.merge(wh_metadata.drop(columns=['speaker_id']), how='inner', on='raw_text').drop_duplicates(subset=['audio_id', 'raw_text'])
#
# union.set_index('audio_id', inplace=True)
#
# train_metadata = union.loc[union.index.intersection(train_metadata['audio_id'])]
# test_metadata = union.loc[union.index.intersection(test_metadata['audio_id'])]
#
# union.to_csv(dataset_path / 'metadata.csv')
# train_metadata.to_csv(dataset_path / 'train_metadata.csv')
# test_metadata.to_csv(dataset_path / 'test_metadata.csv')


# %%
# import shutil
# from pathlib import Path
#
# import pandas as pd
# import tqdm
# metadata = []
#
# output_path = Path('../data/VCTK-Corpus')
# text_path = output_path/'VCTK-Corpus'/'txt'
# input_audio_path = output_path/'VCTK-Corpus'/'wav48'
#
# for text_file in tqdm.tqdm(list(text_path.glob('**/*.txt'))):
#     speaker_id = text_file.stem.split('_')[0]
#     audio_id = text_file.stem + '.wav'
#     with open(text_file, 'r') as f:
#         text = f.read().strip()
#
#     metadata.append(
#         {
#             'speaker_id': speaker_id,
#             'audio_id': audio_id,
#             'text': text,
#         }
#     )
#     audio_src = input_audio_path / speaker_id/ audio_id
#     audio_dst = output_path / 'wavs' / audio_id
#     shutil.move(audio_src, audio_dst)
#
# metadata = pd.DataFrame(metadata)
# metadata.to_csv(output_path/'metadata.csv', index=False)


# %%
import shutil
from pathlib import Path

# import pandas as pd
# import tqdm
# with open('../data/VCTK-Corpus/speaker-info.txt', 'r') as f:
#     speaker_info = f.read()
#
# speaker_info = speaker_info.split('\n')
# columns = speaker_info[0]
# speaker_info = speaker_info[1:]
#
# columns = [column for column in columns.split(' ') if column]
# speaker_info = [[data for data in info.split(' ') if data] for info in speaker_info]
# speaker_info = [info[:-2] + [info[-2] + ' ' + info[-1]] for info in speaker_info]
#
# speaker_info = pd.DataFrame(speaker_info, columns=columns)
# speaker_info.to_csv('../data/VCTK-Corpus/speaker_info.csv', index=False)


# %%
# import string
# output_path = Path('../data/VCTK-Corpus')
# metadata = pd.read_csv(output_path/'metadata.csv')
#
# # chr_to_replace = ['"', "'", ')', '`']
# #
# # for chr in chr_to_replace:
# #     metadata['text'] = metadata['text'].str.replace(chr, '')
# #
# # metadata['text'] = metadata['text'].str.replace('\t', ' ')
# # metadata.to_csv(output_path/'metadata.csv', index=False)
#
# text = metadata['text'].sum()
# print({c for c in text if not c.isalnum()})

# %%
# from sklearn.model_selection import train_test_split
#
# output_path = Path('../data/VCTK-Corpus')
# metadata = pd.read_csv(output_path / 'metadata.csv')
#
# speakers = metadata['speaker_id'].value_counts().iloc[:2].index.tolist()
# speakers = metadata[metadata['speaker_id'].isin(speakers)]
# speakers = speakers.sample(frac=1, random_state=42).groupby('speaker_id').head(1)
#
# metadata = metadata[~metadata['audio_id'].isin(speakers['audio_id'].tolist())]
#
# train, test = train_test_split(metadata, test_size=0.1, stratify=metadata['speaker_id'], random_state=42)
# speakers.to_csv(output_path / 'speakers.csv', index=False)
# train.to_csv(output_path / 'train_metadata.csv', index=False)
# test.to_csv(output_path / 'test_metadata.csv', index=False)


##%
# test = pd.read_csv('../data/VCTK-Corpus/test_metadata.csv')
# speakers_info = pd.read_csv('../data/VCTK-Corpus/speaker_info.csv')
#
# speakers = speakers_info.loc[speakers_info['ACCENTS'].isin(['Indian', 'English', 'American'])]
# speakers = speakers.sample(frac=1, random_state=42)
# speakers = speakers.groupby('ACCENTS').head(1)
# speakers['ID'] = speakers['ID'].apply(lambda x: f'p{x}')
#
# speakers_data = test.loc[test['speaker_id'].isin(speakers['ID'])]
# speakers_data = speakers_data.sample(frac=1, random_state=42)
# speakers_data = speakers_data.groupby('speaker_id').head(1)
# speakers_data = speakers_data.merge(speakers, left_on='speaker_id', right_on='ID')
# speakers_data.to_csv('../data/VCTK-Corpus/speakers_to_sample.csv', index=False)
