import os
import re
import cv2
import keras
import pyttsx3
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# Read images
def readImage(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img/255.
    return img

# Clean data
def process(data):
    clean_data = []
    for word in data.split():
        if len(word) > 1:
            word = word.lower()
            word = re.sub('[^A-za-z]', '', word)
            clean_data.append(word)
    return clean_data

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def prediction(model, image, tokenizer, max_caption_len):
    in_text = 'start'
    for i in range(max_caption_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_caption_len)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def speech_audio(text):
    engine = pyttsx3.init()
    if not pyttsx3.init():
        engine = pyttsx3.init()
        engine.save_to_file(text, 'audio.mp3')
        engine.runAndWait()
        audio_file = open(f"audio.mp3", "rb")
        audio_bytes = audio_file.read()
        st.markdown("## Your audio:")
        st.audio(audio_bytes, format="audio/mp3", start_time=0)
    else :
        engine.save_to_file(text, 'audio.mp3')
        engine.runAndWait()
        audio_file = open(f"audio.mp3", "rb")
        audio_bytes = audio_file.read()
        st.markdown("## Your audio:")
        st.audio(audio_bytes, format="audio/mp3", start_time=0)