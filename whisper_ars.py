import whisper
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os

# File relative to current script
# file_path = os.path.join(os.path.dirname(__file__), "recording", "recording0.wav")
# print(file_path)


AUDIO_PATH = file_path = os.path.join(os.path.dirname(__file__), "recording", "recording0.wav")

def speedch_recognition():
    model = whisper.load_model("medium") # Load model (choose "tiny", "base", "small", "medium", or "large")
    # print("Whisper model loaded")


    # freq = 44100 # Sampling frequency
    # duration = 6 # Recording duration

    # # Start recorder with the given values 
    # # of duration and sample frequency
    # recording = sd.rec(int(duration * freq), 
    #                 samplerate=freq, channels=1)

    # sd.wait() # Record audio for the given number of seconds
    # wv.write(AUDIO_PATH, recording, freq, sampwidth=2) # Convert the NumPy array to audio file


    # audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    # mel = whisper.log_mel_spectrogram(torch.tensor(audio))
    result = model.transcribe(AUDIO_PATH)


    return result["text"]