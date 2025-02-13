from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split


def group_scoring(group, ascending=True):
    # Sort values and reset index to get positions 1 to n
    sorted_group = group.sort_values(ascending=ascending)
    n = len(sorted_group)

    # Create scores from 1 to n (or 1 to 10 if you specifically need 10)
    scores = pd.Series(
        data=range(1, n + 1),
        index=sorted_group.index
    )
    return scores / n


def f3(row):
    cer = row['cer_rank']
    secs = row['secs_rank']
    utmos = row['utmos_rank']

    return 3 / (1 / cer + 1 / secs + 1 / utmos)


def build_dataset(group, metric='f3_rank'):
    sorted_g = group.sort_values(metric)
    good = sorted_g.iloc[1]
    bad = sorted_g.iloc[-2]
    res = pd.Series(
        {
            'audio_id': bad['audio_id'],
            'speaker_id': bad['speaker_id'],
            'text': bad['text'],
            'mel_cond_l': bad['gpt_codes'],
            'mel_cond_w': good['gpt_codes'],
            'l_rank': bad[metric],
            'w_rank': good[metric],
        }
    )
    return res


df = pd.read_parquet('../../data/VCTK-Corpus_gen/vctk_asr_gen.parquet')

# Apply ranking for each metric
df['cer_rank'] = df.groupby(['speaker_id', 'prompt_id'])['cer'].transform(group_scoring)
df['secs_rank'] = df.groupby(['speaker_id', 'prompt_id'])['secs'].transform(partial(group_scoring, ascending=False))
df['utmos_rank'] = df.groupby(['speaker_id', 'prompt_id'])['utmos'].transform(partial(group_scoring, ascending=False))

df['f3_rank'] = df.apply(f3, axis=1)

df.to_parquet('../../data/VCTK-Corpus_gen/vctk_asr_gen.parquet')

dpo_dataset = df.groupby('prompt_id').apply(build_dataset)
dpo_dataset.to_parquet('../../data/VCTK-Corpus_gen/dpo_data.parquet')

# dpo_dataset = pd.read_parquet('../../data/keithito_lj_speech/dpo_data.parquet')
# train_df = pd.read_csv('../../data/keithito_lj_speech/train_metadata.csv')
# test_df = pd.read_csv('../../data/keithito_lj_speech/test_metadata.csv')
#
# train_df['audio_id'] = train_df['audio_id'] + '.wav'
# test_df['audio_id'] = test_df['audio_id'] + '.wav'
#
# # train, test = train_test_split(dpo_dataset, test_size=0.1, stratify=dpo_dataset['speaker_id'], random_state=42)
# train = dpo_dataset[dpo_dataset['audio_id'].isin(train_df['audio_id'])]
# test = dpo_dataset[dpo_dataset['audio_id'].isin(test_df['audio_id'])]
# train.to_parquet('../../data/keithito_lj_speech/dpo_data_train.parquet')
# test.to_parquet('../../data/keithito_lj_speech/dpo_data_test.parquet')
