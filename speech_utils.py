'''
AUTHOR: Peter Ball
email: peter.ball@mail.mcgill.ca
'''
import speech_recognition as sr
import pickle
import json

from nltk.tokenize import word_tokenize
import parselmouth
from collections import Counter

import numpy as np
import math

_COUNTER_PATH = './counter.pickle'
_META_PATH = './meta.json'
_TRANSCRIPT_PATH = './transcript.txt'
_TIME_INTERVAL = 5

_FILLER_WORDS = ['like', 'really', 'right', 'totally']

_MESSAGES = {
	'Anger': "You seem angry. Take some deep breaths and regroup.",
	'Fear': "You are coming across as nervous. Speak slowly and measuredly.",
	'Fillers': "You are using {} a lot. Make every word count!",
	'Speed': "You are speaking at {} words/second, when you normally speak at {} words/second.",
	'No Comment': "You're sounding great!"
}

class Recording(object):
	def __init__(self, recognizer, filepath):
		self.processed = self.process_voice(recognizer, filepath)
		self.transcription = self.processed['transcription']
		self.tokens = word_tokenize(self.transcription)

		self.fillers = self.find_fillers()
		self.num_words = len(self.tokens)

		self.pm_sound = parselmouth.Sound(filepath)
		self.intensity = self.get_intensity()
		self.pitch = self.get_pitch()
		self.audio_length = self.pm_sound.get_total_duration()

	def process_voice(self, recognizer, filepath):
		#THis method takes ideas from https://realpython.com/python-speech-recognition/
		voice = sr.AudioFile(filepath)
		with voice as source:
			recognizer.adjust_for_ambient_noise(source)
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
		fillers = [x for x in self.tokens if x in _FILLER_WORDS]
		
		filler_counts = Counter(fillers)
		response = {
			'Status': False,
			'Filler Counts': None
		}

		if filler_counts:
			response['Status'] = True
			response['Filler Counts'] = filler_counts

		return response

	def get_intensity(self):
		return self.pm_sound.to_intensity().values.T

	def get_pitch(self):
		pitch_obj = self.pm_sound.to_pitch()
		pitch_values = pitch_obj.selected_array['frequency']
		pitch_values[pitch_values==0] = np.nan
		return pitch_values

	def output(self):
		result = {
			'transcription': self.transcription,
			'fillers': self.fillers
		}


class Profile(object):
	def __init__(self, mean_pitch, std_pitch, mean_inten, std_inten, rate_of_speech):
		self.mean_pitch = mean_pitch
		self.std_pitch = std_pitch
		self.mean_inten = mean_inten
		self.std_inten = std_inten
		self.rate_of_speech = rate_of_speech

	def to_json(self, filepath):
		#WARNING: overwrites file at filepath!
		output = {
			'mean_pitch': self.mean_pitch,
			'std_pitch': self.std_pitch,
			'mean_inten': self.mean_inten,
			'std_inten': self.std_inten,
			'rate_of_speech': self.rate_of_speech

		}

		with open(filepath, 'w') as fp:
			json.dump(output, fp)

	def from_json(filepath):

		with open(filepath) as fp:
			result = json.load(fp)

		return(Profile(result['mean_pitch'],
					   result['std_pitch'],
					   result['mean_inten'], 
					   result['std_inten'],
					   result['rate_of_speech']))

class Prediction(object):
	def __init__(self, message, score):
		self.message = message
		self.score = score

	def to_json(self):
		output = {
			'message': self.message,
			'score': self.score
		}

		return json.dumps(output)

	def __str__(self):
		return("Message: {0}, Score: {1}".format(self.message, self.score))

class Recorder(object):
	def __init__(self):
		self.recordings = []
		self.recognizer = sr.Recognizer()

	def gen_profile(self, filepath):
		rec = Recording(self.recognizer, filepath)

		mean_pitch = np.nanmean(rec.pitch)
		std_pitch = np.nanstd(rec.pitch)

		mean_inten = np.mean(rec.intensity)
		std_inten = np.std(rec.intensity)

		rate_of_speech = rec.num_words/rec.audio_length
		return Profile(mean_pitch, std_pitch, mean_inten, std_inten, rate_of_speech)



	def add_recording(self, filepath):
		new_recording = Recording(self.recognizer, filepath)
		self.recordings.append(new_recording)

	def analyse(self, profile):
		pitches = []
		intensities = []
		for recording in self.recordings:

			pitches.append(recording.pitch)
			intensities.append(recording.intensity)

		predictions = []
		best_predict = Prediction(_MESSAGES['No Comment'], 0)

		#Only make recomendations once they've been talking for a bit
		#Only make recomendations based on recent time window
		#Recommendation weighting is pretty hacky!
		if len(self.recordings) > 2:
			rec_objs = self.recordings[-3:]
			freq_dist = np.concatenate((pitches[-3], pitches[-2], pitches[-1]))
			intensity_dist = np.concatenate((intensities[-3], intensities[-2], intensities[-1]))

			freq_mean = np.nanmean(freq_dist)
			intensity_mean = np.mean(intensity_dist)
			wordcount_mean = np.mean([x.num_words for x in rec_objs])
			audio_length_mean = np.mean([x.audio_length for x in rec_objs])

			print(freq_mean)
			print(intensity_mean)
			print(profile.mean_pitch)
			print(profile.mean_inten)

			filler_totals = Counter()
			for recording in rec_objs:
				if recording.fillers['Status']:
					filler_totals += recording.fillers['Filler Counts']

			wc_modifier = math.log(abs(profile.rate_of_speech-wordcount_mean), 200)
			filler_score = math.log(sum(list(filler_totals.values())) / audio_length_mean)

			rlyhigh_freq = False
			high_freq = False
			low_freq = False
			high_inten = False
			low_inten = False

			emotion = ""
			#if the average frequency is more than a standard deviation away from the mean:
			if freq_mean > profile.mean_pitch + 1.5*profile.std_pitch:
				rlyhigh_freq = True
			elif freq_mean > profile.mean_pitch + 1*profile.std_pitch:
				high_freq = True
			elif freq_mean < profile.mean_pitch - 1.5*profile.std_pitch:
				low_freq = True

			# If the average intensity is more than a standard deviation away from the mean:
			if intensity_mean > profile.mean_inten + profile.std_inten:
				high_inten = True
			if intensity_mean < profile.mean_inten - profile.std_inten:
				low_inten = True

			if high_freq:
				if rlyhigh_freq and high_inten:
					predictions.append(Prediction(_MESSAGES['Anger'], 0.4 + wc_modifier))
				else:
					predictions.append(Prediction(_MESSAGES['Fear'], 0.4 + wc_modifier))

			predictions.append(Prediction(_MESSAGES['Speed'].format(wordcount_mean/audio_length_mean, profile.rate_of_speech), wc_modifier))
			if filler_totals:
				predictions.append(Prediction(_MESSAGES['Fillers'].format(filler_totals.most_common(1)[0]), filler_score))

			for prediction in predictions:
				if prediction.score > best_predict.score:
					best_predict = prediction
		print([str(x) for x in predictions])
		return best_predict.to_json()




rec = Recorder()
prof = rec.gen_profile('../test/peter_profile.wav')
prof.to_json('./profiles/Peter.json')
prof = Profile.from_json('./profiles/Neshma.json')

rec.add_recording('../test/na1.wav')
rec.add_recording('../test/na2.wav')
rec.add_recording('../test/na3.wav')


print(rec.analyse(prof))