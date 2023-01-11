import pandas as pd
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import tensorflow as tf
from pathlib import Path
import os
from tensorflow import keras

def extract_feature(filename):
  audio, sample_rate = librosa.load(filename)
  #audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
  mfcc_scaled = np.mean(mfcc.T, axis=0)
  return mfcc_scaled

def predict(path, model):
    audio = np.array([extract_feature(path)])
    classid = np.argmax(model.predict(audio)[0])
    df = pd.read_csv('UrbanSound8K.csv')
    classes = df.groupby('classID')['class'].unique()
    #classes
    return classes[classid][0]
    #print('Class predicted :',classes[classid][0],'\n\n')
    #return ipd.Audio(path)

def predictClass(path):
    #model = tf.keras.models.load_model('/models/my_model')
    #model = tf.keras.models.load_model('mode_ann.h5')
    h5_model = tf.keras.models.load_model('model_ann.h5')
    #acc = str(h5_model.count_params())
    #return acc 
    in_fpath = Path(path.replace('"', "").replace("'", ""))
    return predict(in_fpath, h5_model)

#if __name__ == "__main__":
#  path = "dog.wav"
#  predictClass(path)