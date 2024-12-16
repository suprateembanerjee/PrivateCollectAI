from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
from transformers import pipeline
import numpy as np

def diarize(audio_data,
            model='pyannote/speaker-diarization-3.1',
            # fname='sample.wav',
            num_speakers=2):
    pipeline = Pipeline.from_pretrained(model)
    conversation = pipeline(audio_data, num_speakers=num_speakers)
    segments = {}

    for segment in conversation.itertracks(yield_label=True):
        duration, _, speaker = segment
        segments[speaker] = segments.get(speaker, []) + list(duration)
    
    return segments

def consolidate(audio_data,
                segments):
    
    audio = AudioSegment.from_file(audio_data)
    consolidated = {}

    for speaker in segments:
        merged_audio = AudioSegment.silent(duration=0)
        i = 0
        while i < len(segments[speaker]):
            merged_audio += audio[int(segments[speaker][i] * 1000): int(segments[speaker][i + 1] * 1000)]
            i += 2
        consolidated[speaker] = merged_audio
    
    return consolidated

def transcribe(audio_data):

    # audio_arr = np.array(audio_data.get_array_of_samples(), dtype=np.float32)
    
    whisper = pipeline('automatic-speech-recognition', 
                       'openai/whisper-large-v3', 
                       torch_dtype=torch.float16, 
                       device='mps')
    
    transcription = whisper('sample.wav')

    return transcription['text']