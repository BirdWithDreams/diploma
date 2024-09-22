import pandas as pd


def main():
    train_meta = []
    with open('../data/keithito_lj_speech/train_metadata.txt', 'r') as f:
        for line in f.readlines():
            item = line.split('|')
            train_meta.append({
                'audio_id': item[0].split('.')[0],
                'raw_text': item[1],
                'speaker_id': 'lg_speaker',
            })

    train_meta = pd.DataFrame(train_meta)
    train_meta.to_csv('../data/keithito_lj_speech/train_metadata.csv')

    test_meta = []
    with open('../data/keithito_lj_speech/test_metadata.txt', 'r') as f:
        for line in f.readlines():
            item = line.split('|')
            test_meta.append({
                'audio_id': item[0].split('.')[0],
                'raw_text': item[1],
                'speaker_id': 'lg_speaker',
            })

    test_meta = pd.DataFrame(test_meta)
    test_meta.to_csv('../data/keithito_lj_speech/test_metadata.csv')

    meta_df = pd.concat([train_meta, test_meta])
    meta_df.to_csv('../data/keithito_lj_speech/metadata.csv')


if __name__ == '__main__':
    main()
