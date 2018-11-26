import audio_preprocessing
import os
import gc
import torch
import numpy as np
CHUNK_SIZE = 100
# 30 sec eqvivalent  
FILE_LENGTH_FOR_TESTING = 1320000
NUMBER_OF_RAW_TRACKS = 72


def get_chunks(source):
	'''
	:params data set path
	:returns a tuple or two tensors for each song
	'''
	songs = []
	for root, dirnames, filenames in os.walk(source):
		song_folders = sorted(dirnames)
		break;
	for song_folder in song_folders:
		for root, dirnames, filenames in os.walk(song_folder):
			if filter(lambda x:x[-4:]==".mp3",filenames):
				mixed_song = "./"+root+"/"+filter(lambda x:x[-4:]==".mp3",filenames)[0]
			if sorted(filter(lambda x:x[-4:]==".wav",filenames)):
				audio_files = sorted(filter(lambda x:x[-4:]==".wav",filenames))
		chunked_data = []
		for audio_file in audio_files:
			song_data = []
			audio_data = audio_preprocessing.get_audio_data("./"+root+"/"+audio_file)
			for i in range(FILE_LENGTH_FOR_TESTING/CHUNK_SIZE):
				song_data.append(audio_data[100*i:100*(i+1)])
			chunked_data.append(song_data)
		yield (torch.tensor(np.array(chunked_data),dtype = torch.float32),torch.tensor(np.array(audio_preprocessing.get_audio_data(mixed_song)[100*i:100*(i+1)]),dtype = torch.float32))

if __name__ == '__main__':
	gc.collect()
	x = get_chunks("./")
	print x.next()

