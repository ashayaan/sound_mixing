# import os
import librosa
import config

# import numpy as np
# import soundfile as sf
# import scipy.io.wavfile as wavfile


def get_audio_data(audio_file_path):
    """
    takes in a audio file path, parses audio file based on type and returns a numpy array
    :param audio_file_path:
    :return audio_data: numpy array
    """
    audio_data = None

    try:
        audio_data, sample_rate = librosa.load(audio_file_path, sr=config.SAMPLE_RATE)
    except Exception:
        import traceback
        error = traceback.format_exc()
        print "ERROR OCCURRED : ", error

    return audio_data


def pad_audio_data(audio_data):
    """
    :param audio_data: the numpy audio data of a raw audio file
    :param max_length: max length to which an audio file should be padded
    :return padded audio_data:
    """
    audio_data = librosa.util.fix_length(audio_data, config.MAX_LENGTH)
    return audio_data


if __name__ == "__main__":
    audio_file_path = "sample_wav_file.wav"
    audio_data = get_audio_data(audio_file_path)
    pad_audio_data(audio_data)