import audio_preprocessing
import os

def find_max_tracks_type(type_,sub_type_,source):
	max_tracks = 0
	for root, dirnames, filenames in os.walk(source):
		if len(filter(lambda x:x[3:5] == sub_type_ and x[:2]==type_,filenames)) > max_tracks:
			max_tracks = len(filter(lambda x:x[3:5] == sub_type_ and x[:2]==type_,filenames))
	return max_tracks

def find_max_length(source):
	max_length = 0
	for root, dirnames, filenames in os.walk(source):
		audio_files = filter(lambda x:x[-4:]==".wav",filenames)
		for audio_file in audio_files:
			lenght_of_track = len(audio_preprocessing.get_audio_data(os.path.join(root,audio_file)))
			if lenght_of_track > max_length:
				max_length = lenght_of_track
	return max_length

def make_all_file_lengths_equal(source):
	for root, dirnames, filenames in os.walk(source):
		audio_files = filter(lambda x:x[-4:]==".wav",filenames)
		for audio_file in audio_files:
			audio_preprocessing.change_length_audio_file(os.path.join(root,audio_file))

def make_number_of_files_equal(source,types):
	for type_ in types.keys():
		instruments = map(lambda x: str(x).zfill(2),range(types[type_]+1))
		for i in instruments:
			max_tracks = find_max_tracks_type(str(type_).zfill(2),i,"./")
			for root, dirnames, filenames in os.walk(source):
				if len(filter(lambda x:x[-4:]==".wav",filenames)) > 0:
					number_of_tracks = len(filter(lambda x:x[3:5] == i and x[:2]==str(type_).zfill(2),filenames))
					if number_of_tracks < max_tracks:
						for j in range(max_tracks - number_of_tracks):
							audio_preprocessing.create_empty_track(os.path.join(root,str(type_).zfill(2)+"_"+i+"_"+str(number_of_tracks+j+1).zfill(2)+"_empty.wav"))


if __name__ == '__main__':
	# print find_max_length("./")
	types = {1:12,2:4,3:4,4:2,5:1}
	make_all_file_lengths_equal("./")
	make_number_of_files_equal("./",types)

