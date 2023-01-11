import streamlit as st
import soundfile as sf
import os
import glob
from pathlib import Path
import classPredictor 
#import tensorflow as tf
#from tensorflow import keras
from helper import read_audio, record, save_record


#model_load = st.text("Loading model")

#h5_model = tf.keras.models.load_model('model_ann.h5')

#model_load.text("Loaded model")

#Recording voice
st.header("Record")
filename = st.text_input("Choos4e a filename: ")

if st.button(f"Click to Record"):
    if filename == "":
        st.warning("Choose a filename")
    else:
        record_state = st.text("Recording")
        duration = 3 #seconds
        fs = 48000
        myrecording = record(duration, fs)
        record_state.text(f"Saving sample as {filename}.wav")

        path_myrecording = f"./samples/{filename}.wav"

        save_record(path_myrecording, myrecording, fs)
        record_state.text(f"Saved sample as {filename}.wav")

        st.audio(read_audio(path_myrecording))
        
        predClass = classPredictor.predictClass(path_myrecording)
        st.write(predClass)

#Loading audio file
audio_folder = "samples"
filenames = glob.glob(os.path.join(audio_folder, "*.wav"))
selected_filename = st.selectbox("Select a file", filenames)

if selected_filename is not None:
    #in_fpath = Path(selected_filename.replace('"', "").replace("'", ""))
    #in_fpath = Path(selected_filename.replace('"', "").replace("'", ""))
    predClass = classPredictor.predictClass(selected_filename)
    st.write(predClass)