import audio_preprocessing
import os
# audio_preprocessing.get_audio_duration()

def unzip_all(source):
	for root, dirnames, filenames in os.walk(source):
		for filename in filenames:
			if filename[-4:] == ".zip":
				os.system("unzip -d " + root + " " + os.path.join(root,filename))
def find_max_tracks(source):
	max_tracks = 0
	for root, dirnames, filenames in os.walk(source):
		if len(filter(lambda x:x[-4:]==".wav",filenames)) > max_tracks:
			max_tracks = len(filter(lambda x:x[-4:]==".wav",filenames))
	return max_tracks
def find_max_length(source):
	max_length = 0
	max_length_np = 0
	for root, dirnames, filenames in os.walk(source):
		audio_files = filter(lambda x:x[-4:]==".wav",filenames)
		for audio_file in audio_files:
			try:
				lenght_of_track = audio_preprocessing.get_audio_duration(os.path.join(root,audio_file))
			except AssertionError:
				print "Empty file lenght_of_track cannot be found"
				print os.path.join(root,audio_file)			
			try:
				lenght_of_track_np = len(audio_preprocessing.get_audio_data(os.path.join(root,audio_file)))
			except TypeError:
				print "Empty file lenght_of_track_np cannot be found"
				print os.path.join(root,audio_file)
			if lenght_of_track > max_length:
				max_length = lenght_of_track
			if lenght_of_track_np > max_length_np:
				max_length_np = lenght_of_track_np
		# print max_length,max_length_np
	return max_length,max_length_np
 
if __name__ == '__main__':
	# unzip_all("./")
	# print find_max_tracks("./")
	print find_max_length("./")