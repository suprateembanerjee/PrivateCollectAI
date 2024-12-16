import streamlit as st
import os
import io
from pydub import AudioSegment
from io import BytesIO
from utils import diarize, consolidate, transcribe

# Function to play audio from a WAV file
def play_audio(wav_file):
    audio = AudioSegment.from_wav(wav_file)
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)
    return audio_bytes

# Streamlit app
def main():
    st.title('PrivateCollectAI')
    
    # File uploader widget
    uploaded_file = st.file_uploader('Upload original conversation', type='wav')
    
    if uploaded_file is not None:
        # Play the uploaded audio file
        st.audio(uploaded_file, format='audio/wav')
        st.success('Conversation uploaded!')
        audio_data = BytesIO(uploaded_file.getbuffer())
        segments = diarize(audio_data=audio_data)
        consolidated = consolidate(audio_data=audio_data, segments=segments)

        st.write('Diarized:')
        for speaker in consolidated:
            audio_bytes = io.BytesIO()
            consolidated[speaker].export(audio_bytes, format='wav')
            audio_bytes.seek(0)
            st.audio(audio_bytes, format='audio/wav')
        
        st.write(f'Transcription:\n{transcribe(audio_data)}')
        
        
        

if __name__ == '__main__':
    main()