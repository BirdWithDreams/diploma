import re
from pathlib import Path

import pandas as pd

generated_samples_path = Path('../data/my_tests/audio/good_fn_sample')
prompts = pd.read_csv('../data/my_tests/prompts.csv', index_col=0)

speakers_paths = {
    speaker_path.stem: speaker_path.absolute() for speaker_path in Path('../data/speakers').glob('*.wav')
}


def parse_wav_name(filename):
    pattern = r'\((.+?)\)_(.+?)_(\d+)\.wav'

    # Try to match the pattern
    match = re.match(pattern, filename)

    if match:
        # If there's a match, extract the groups
        speaker_name = match.group(1)
        id_ = match.group(2)
        gen_i = int(match.group(3))  # Convert to integer

        return speaker_name, id_, gen_i
    else:
        # If there's no match, return None for all values
        return None, None, None


custom_generated_sentences = []
for file in generated_samples_path.glob('*.wav'):
    speaker_name, id_, gen_i = parse_wav_name(file.name)
    custom_generated_sentences.append(
        {
            'speaker_reference': str(speakers_paths[speaker_name]),
            'text': prompts.loc[int(id_), 'prompt'],
            'generated_wav': str(file.absolute()),
            'language': 'en',
        }
    )

pd.DataFrame(custom_generated_sentences).to_csv('../data/my_tests/audio/good_fn_sample/custom_generated_sentences.csv',
                                                index=False)
