import librosa
import config

import numpy as np

# alternnate modules for audio processing
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
        audio_data, sample_rate = librosa.load(audio_file_path, duration = 60,sr=config.SAMPLE_RATE)
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
    print audio_data.shape
    return audio_data


def create_empty_track(output_file_path):
    """
    :param output_file_path: path to which the empty audio file should be written
    :return audio_data: numpy array
    """
    audio_data = np.zeros([config.MAX_LENGTH, ])
    librosa.output.write_wav(output_file_path, audio_data, config.SAMPLE_RATE)
    change_length_audio_file(output_file_path)
    return audio_data

def change_length_audio_file(audio_file_path):
    """
    :param audio_file_path:
    :return audio_duration:
    """
    audio_data = get_audio_data(audio_file_path)
    audio_data = pad_audio_data(audio_data)
    librosa.output.write_wav(audio_file_path, audio_data, config.SAMPLE_RATE)
    return audio_data

def get_audio_duration(audio_file_path):
    """
    :param audio_file_path:
    :return audio_duration:
    """
    audio_data = get_audio_data(audio_file_path)
    audio_duration = librosa.core.get_duration(audio_data, sr=config.SAMPLE_RATE)
    return audio_duration


if __name__ == "__main__":
    audio_file_path = "dataset/sample_audio_files/sample_wav_file.wav"
    audio_data = get_audio_data(audio_file_path)
    pad_audio_data(audio_data)
