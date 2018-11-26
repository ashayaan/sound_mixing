import audio_preprocessing
import os
import gc
import torch
import numpy as np
import argparse

CHUNK_SIZE = 500
# 30 sec eqvivalent  
FILE_LENGTH_FOR_TESTING = 1320000
NUMBER_OF_RAW_TRACKS = 72


def get_chunked_songs(source):
	'''
	:params data set path
	:returns a tuple or two tensors for each song
	'''
	songs = []
	for root, dirnames, filenames in os.walk(source):
		song_folders = sorted(dirnames)
		break;
	for song_folder in song_folders:
		print song_folder
		for root, dirnames, filenames in os.walk(song_folder):
			if filter(lambda x:x[-4:]==".mp3",filenames):
				mixed_song = "./"+root+"/"+filter(lambda x:x[-4:]==".mp3",filenames)[0]
			if sorted(filter(lambda x:x[-4:]==".wav",filenames)):
				audio_files = sorted(filter(lambda x:x[-4:]==".wav",filenames))
		raw_song_chunked_data = []
		for audio_file in audio_files:
			raw_song_data = []
			audio_data = audio_preprocessing.get_audio_data("./"+root+"/"+audio_file)
			for i in range(FILE_LENGTH_FOR_TESTING/CHUNK_SIZE):
				raw_song_data.append(audio_data[CHUNK_SIZE*i:CHUNK_SIZE*(i+1)])
			raw_song_chunked_data.append(raw_song_data)
		mixed_song_data_chunked = []
		mixed_song_data = audio_preprocessing.get_audio_data(mixed_song)
		for i in range(FILE_LENGTH_FOR_TESTING/CHUNK_SIZE):
			mixed_song_data_chunked.append(mixed_song_data[CHUNK_SIZE*i:CHUNK_SIZE*(i+1)])
		yield (torch.tensor(np.array(raw_song_chunked_data), dtype=torch.float, requires_grad=False),torch.tensor(np.array([mixed_song_data_chunked]), dtype=torch.float, requires_grad=False))


if __name__ == '__main__':
	gc.collect()
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="./", help="path to the dataset")
	args = parser.parse_args()
	x = get_chunked_songs(args.datapath)
