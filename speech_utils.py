'''
AUTHOR: Peter Ball
email: peter.ball@mail.mcgill.ca
'''
import speech_recognition as sr
import csvS
import pickle

from nltk.tokenize import word_tokenize
import parselmouth

_COUNTER_PATH = './counter.pickle'
_META_PATH = './meta.json'
_TRANSCRIPT_PATH = './transcript.txt'

class Recording(object):
	def __init__(self, recognizer, filepath):
		self.processed = process_voice(recognizer, filepath)
		self.transcription = processed['transcription']
		self.tokens = word_tokenize(transcription)

		self.fillers = find_fillers()
		self.num_words = len(tokens)

		self.pm_sound = parselmouth.Sound(filepath)
		self.avg_intensity = get_avg_intensity()
		self.avg_pitch = get_avg_pitch()

	def process_voice(self, recognizer, filepath):
		#THis method takes ideas from https://realpython.com/python-speech-recognition/
		recognizer.adjust_for_ambient_noise(source)
		source = sr.AudioFile(filepath)
		audio = recognizer.record(source)

		response = {
			"success": True,
			"error": None,
			"transcription": None
		}

		try:
			response["transcription"] = recognizer.recognize_google(audio)
		except sr.RequestError:
			#API unreachable
			response["success"] = False
			response["error"] = "API unavailable"
		except sr.UnknownValueError:
			#Speech uninteligible
			response["success"] = False
			response["error"] = "Unable to recognize speech"

		return response

	def find_fillers(self):
		fillers = [x for x in self.tokens if x in filler_words]
		
		filler_counts = Counter(fillers)
		response = {
			'Status': False
			'Filler Counts': None
		}

		if filler_counts:
			response['Status'] = True
			response['Filler Counts'] = filler_counts

		return response

	def get_avg_intensity(self):
		intensity = self.pm_sound.to_intensity()

		return intensity.get_average()

	def get_avg_pitch(self):
		pitch = self.pm_sound.to_pitch()
		pitch_array = pitch.selected_array['frequency']

		return np.mean(pitch_array)

	def output(self):
		result = {
			'transcription' = self.transcription
			fillers = self.fillers
		}

class Recorder(object):
	def __init__(self):
		self.recordings = []
		self.recognizer = sr.Recognizer()

		analyse()

		def add_recording(self, filepath):
			new_recording = Recording(self.recognizer, filepath)
			self.recordings.append(new_recording)
