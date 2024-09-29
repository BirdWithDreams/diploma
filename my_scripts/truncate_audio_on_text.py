import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from pydub import AudioSegment
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_whisper_model(model_name="openai/whisper-medium"):
    """Load Whisper model and processor from Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    return processor, model, device


def transcribe_audio_with_timestamps(audio_path, processor, model, device):
    """Transcribe audio file using Whisper model and return transcription with timestamps."""
    # Load audio
    # dataset = load_dataset("audio", data_files={"audio": audio_path})
    # audio = dataset["audio"][0]["array"]
    audio, sr = librosa.load(audio_path)
    # Process audio
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)

    # Generate tokens with timestamps
    with torch.no_grad():
        generated_tokens = model.generate(input_features, return_timestamps=True)

    # Decode tokens
    transcription = processor.batch_decode(generated_tokens, skip_special_tokens=True)
    return transcription[0]


def get_truncation_point(transcription, max_chars=250):
    """Find the truncation point that includes as many full sentences as possible within the character limit."""
    current_length = 0
    last_sentence_end = 0

    for chunk in transcription['chunks']:
        chunk_text = chunk['text'].strip()
        if current_length + len(chunk_text) > max_chars:
            break

        current_length += len(chunk_text) + 1  # +1 for space between chunks
        if chunk_text.endswith(('.', '!', '?')):
            last_sentence_end = chunk['timestamp'][-1]

    return last_sentence_end if last_sentence_end > 0 else transcription['chunks'][-1]['timestamp'][-1]


def truncate_audio(audio_path, end_time):
    """Truncate audio file up to the specified end time."""
    try:
        audio = AudioSegment.from_file(audio_path)
        truncated_audio = audio[:int(end_time * 1000)]  # Convert to milliseconds
        return truncated_audio
    except Exception as e:
        logging.error(f"Error truncating audio: {str(e)}")
        return None


def process_audio_file(input_path, output_path, processor, model, device):
    """Process a single audio file: transcribe, find truncation point, and truncate."""
    try:
        # Transcribe audio with timestamps
        transcription = transcribe_audio_with_timestamps(input_path, processor, model, device)

        if not transcription:
            logging.warning(f"No transcription produced for {input_path}")
            return ""

        # Find the truncation point
        end_time = get_truncation_point(transcription)

        # Truncate audio
        truncated_audio = truncate_audio(input_path, end_time)

        if truncated_audio is None:
            logging.warning(f"Failed to truncate audio for {input_path}")
            return ""

        # Save truncated audio
        truncated_audio.export(output_path, format="wav")

        # Prepare transcription for logging or further use
        final_transcription = " ".join(
            chunk['text'] for chunk in transcription['chunks'] if chunk['timestamp'][-1] <= end_time)
        return final_transcription.strip()
    except Exception as e:
        logging.error(f"Error processing file {input_path}: {str(e)}")
        return ""


def main():
    # Load Whisper model and processor
    processor, model, device = load_whisper_model()

    # Directory containing audio files
    input_directory = "../data/facebook_voxpopuli/wavs"
    output_directory = "../data/facebook_voxpopuli/trim_wavs"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each audio file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith((".wav", ".mp3", ".flac")):  # Add more audio formats if needed
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"truncated_{filename}")
            transcription = process_audio_file(input_path, output_path, processor, model, device)
            if transcription:
                logging.info(f"Processed: {filename}")
                logging.info(f"Transcription ({len(transcription)} chars): {transcription}")
            else:
                logging.warning(f"Failed to process: {filename}")


if __name__ == "__main__":
    main()