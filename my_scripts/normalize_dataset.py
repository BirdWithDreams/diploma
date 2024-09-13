import csv
import os

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def reformat_dataset(output_dir):
    # Load the dataset
    dataset = load_dataset('openslr/librispeech_asr', cache_dir='../data', split='train[:10%]+valid')

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # Prepare metadata file
    metadata_path = os.path.join(output_dir, "metadata.txt")

    with open(metadata_path, 'w', newline='', encoding='utf-8') as metadata_file:
        writer = csv.writer(metadata_file, delimiter='|')

        # Iterate through the dataset
        for item in tqdm(dataset['train'], desc="Processing items"):
            # Generate a filename for the audio
            audio_filename = f"{item['id']}.wav"

            # Save the audio file
            audio_path = os.path.join(wavs_dir, audio_filename)
            sf.write(audio_path, item['audio']['array'], item['audio']['sampling_rate'])

            # Write metadata
            writer.writerow([
                audio_filename.replace('.wav', ''),
                item['text'],
                item['normalized_text']
            ])

    print(f"Dataset reformatted and saved to {output_dir}")


# Usage
output_directory = "../data/openslr_librispeech_asr"
reformat_dataset(output_directory)
