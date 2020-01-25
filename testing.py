from flask import Flask, request, jsonify
from speech_utils import Recorder, Recording
from speech_recognition import recognizer

# initialize Flask app
@app.route('/')
app = Flask(__name__)


if __name__ == '__main__': 
    app.run(debug = True)

def outcome(filepath):
    if(request.method == 'GET'):
        recorder = Recorder()
        recording = Recording(recognizer, filepath)
        print(recording)

        return recording


