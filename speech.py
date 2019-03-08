#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:50:46 2019

@author: linuxlite
"""

import numpy as np
import pandas as pd

import os
import subprocess 
os.getcwd()
os.chdir('/home/linuxlite/temp/')

#getting video and audio files 
"""
video_file=raw_input("please input the location of vidoe file")
audio_file=raw_input("please input the location of audio file")
bitrate_to_be_output=
"""
import os
import wave
import pylab
def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('spectrogram.png')
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
graph_spectrogram('harvard.wav')

#Data visualising 
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('harvard.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#converting video to audio 
#bitrate is 42000
"""
we can use raw_input the file 
using two output speakers with one channel
using the frequncy of 160k hertz
"""

command="ffmpeg -i /home/linuxlite/temp/test1.mp4 -ab 160k -ac 2 -ar 42000 -vn /home/linuxlite/temp/audio24.wav"

subprocess.call(command,shell=True)

#processing audio to text
import speech_recognition

rec_obj=speech_recognition.Recognizer()
audio_file_wav = speech_recognition.AudioFile('audio24.wav')

with audio_file_wav as source:
    audio=rec_obj.record(source,duration=4)

#using google api 
output_text=rec_obj.recognize_google(audio)


print(output_text)

#converting to file 
text_file = open("Output.txt", "w")
text_file.write("%s" %output_text)
text_file.close()


import nltk

def graph():
  f = open("Output.txt", "r")
  inputfile = f.read()
  tokens = nltk.tokenize.word_tokenize(inputfile)
  fd = nltk.FreqDist(tokens)
  fd.plot(30,cumulative=False)
graph()

############




#making the bag of model
"""
dataset=pd.read_csv('output.txt',delimiter='\t',quoting=3)
import re 
notes=re.sub('[^a-zA-Z0-9]',)`
notes=notes.lower()

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

word_list=[]
for i in xrange(0,1000):
    x=re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    x=x.lower()
    x=x.split()
    ps=PorterStemmer()
    x=[ps.stem(word) for word in x if not word in set(stopwords.words('english'))]
    x=' '.join(x)
    word_list.append(x)

#bag of model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(word_list).toarray()
Y=dataset.iloc[:,1].values

#spliting 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#naive bayes 
from sklearn.naive_bayes import GaussianNB
naive_object=GaussianNB()
naive_object.fit(x_train,y_train)

y_pred=naive_object.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#by logistics regression
from sklearn.linear_model import LogisticRegression
log_regressor=LogisticRegression()
log_regressor.fit(x_train,y_train)

y_pred_regressor=log_regressor.predict(x_test)
cm_log=confusion_matrix(y_test,y_pred_regressor)

#by support vector classifier
from sklearn.svm import SVC
svc_object=SVC(kernel='rbf',random_state=0)
svc_object.fit(x_train,y_train)
y_pred_svc=svc_object.predict(x_test)

cm_svm=confusion_matrix(y_test,y_pred_svc)

#decision tree classifier 
from sklearn.tree import DecisionTreeClassifier
decision_object=DecisionTreeClassifier(criterion='entropy',random_state=0)
decision_object.fit(x_train,y_train)

y_pred_decision=decision_object.predict(x_test)

cm_decision=confusion_matrix(y_test,y_pred_decision)
"""

