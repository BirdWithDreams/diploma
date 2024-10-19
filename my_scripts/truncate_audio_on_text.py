import tqdm
from faster_whisper import WhisperModel
from pydub import AudioSegment
import os


def load_whisper_model(model_size="medium", device="cuda", compute_type="int8"):
    """Load Whisper model using faster-whisper."""
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


def transcribe_audio_with_timestamps(audio_path, model):
    """Transcribe audio file using Whisper model and return segments with timestamps."""
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    return list(segments)


def get_truncation_point(segments, max_chars=200):
    """Find the truncation point that includes as many full sentences as possible within the character limit."""
    current_length = 0
    last_sentence_end = 0

    for segment in segments:
        segment_text = segment.text.strip()
        if current_length + len(segment_text) > max_chars:
            break

        current_length += len(segment_text) + 1  # +1 for space between segments
        if segment_text.endswith(('.', '!', '?')):
            last_sentence_end = segment.end

    return last_sentence_end if last_sentence_end > 0 else segment.end


def truncate_audio(audio_path, end_time):
    """Truncate audio file up to the specified end time."""
    audio = AudioSegment.from_file(audio_path)
    truncated_audio = audio[:int(end_time * 1000)]  # Convert to milliseconds
    return truncated_audio


def process_audio_file(input_path, output_path, model):
    """Process a single audio file: transcribe, find truncation point, and truncate."""

    if os.path.exists(output_path):
        return None

    # Transcribe audio with timestamps
    try:
        segments = transcribe_audio_with_timestamps(input_path, model)
        if len(segments) == 0:
            return None

    except:
        return None

    # Find the truncation point
    end_time = get_truncation_point(segments)

    # Truncate audio
    truncated_audio = truncate_audio(input_path, end_time)

    # Save truncated audio
    truncated_audio.export(output_path, format="wav")

    # Prepare transcription for logging or further use
    transcription = " ".join(segment.text for segment in segments if segment.end <= end_time)
    return transcription.strip()


def main():
    # Load Whisper model
    model = load_whisper_model()

    # Directory containing audio files
    input_directory = "../data/facebook_voxpopuli/wavs"
    output_directory = "../data/facebook_voxpopuli/trim_wavs"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each audio file in the input directory
    for filename in tqdm.tqdm(os.listdir(input_directory)):
        if filename.endswith((".wav", ".mp3", ".flac")):  # Add more audio formats if needed
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            transcription = process_audio_file(input_path, output_path, model)
            print(f"Processed: {filename}")
            if transcription:
                print(f"Transcription ({len(transcription)} chars): {transcription}")


if __name__ == "__main__":
    main()