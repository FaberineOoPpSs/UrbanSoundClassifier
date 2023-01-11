import streamlit as st
#from classPredictor import predictClass
from PIL import Image
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
#import IPython.display as ipd
import tensorflow as tf
from tensorflow import keras


def extract_feature(filename):
  sound_name = f'audio_files/{filename}'
  audio, sample_rate = librosa.load(sound_name)
  #audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  mfcc_scaled = np.mean(mfcc.T, axis=0)
  return mfcc_scaled

def predict(path, model):
    audio = np.array([extract_feature(path)])
    classid = np.argmax(model.predict(audio)[0])
    df = pd.read_csv('UrbanSound8k.csv')
    classes = df.groupby('classID')['class'].unique()
    #classes
    return classes[classid][0]

# save sound file, uploaded before, in a folder
def save_file(sound_file):
    
    # save your sound file in the right folder by following the path 
    with open(os.path.join('audio_files/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    
    return sound_file.name


# if you have chosen prediction in the sidebar
def choice_prediction():
    st.write('# Prediction')
    st.write('### Choose audio file in .wav format')
    
    # upload sound
    uploaded_file = st.file_uploader(' ', type='wav')
    
    if uploaded_file is not None:
            
        # view details
        file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        st.write(file_details)
        
        # read and play the audio file
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
      
        # save_file function
        save_file(uploaded_file)

        # define the filename
        sound = uploaded_file.name
        
        
        
        st.write('### Classification results')



        model = tf.keras.models.load_model('model_ann.h5') 


        # if you select the predict button
        if st.button('Predict'):
            st.write("Sound is: ", str(predict(sound, model)))

    else:
        st.write('The file has not been uploaded yet')
    
    return
        
        
        
# main
if __name__ == '__main__':
    
    st.image(Image.open('images.jpg'), width=200)
    st.write('___')
    
    # create a sidebar
    st.sidebar.title('Urban sounds classification')
    select = st.sidebar.selectbox('', ['Select Prediction', 'Prediction'], key='1')
    st.sidebar.write(select)
    
    # if sidebar selection is "Prediction"
    if select=='Prediction':
        # choice_prediction function
        choice_prediction()
    
    # else: stay on the home page
    else:
        st.write('# Sounds')
        