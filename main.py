import os
import pickle
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from textwrap import wrap
import re
import keras
from PIL import Image
import streamlit as st
import easygui
import pyttsx3

# Read caption
img_id_caption = pd.read_csv("flickr8k/captions.txt", sep=',')
print(len(img_id_caption))


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


img_id_caption['cleaned_caption'] = img_id_caption['caption'].apply(
    lambda x: 'start '+' '.join(process(x)) + ' end')
all_captions = img_id_caption['cleaned_caption'].to_list()
print(len(all_captions))

img_id_caption.drop(columns=['caption'], inplace=True)
# Convert DataFrame to dictionary with lists for duplicate keys
caption_dict = {k: v.tolist()
                for k, v in img_id_caption.groupby('image')['cleaned_caption']}

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
voc_size = len(tokenizer.word_index)+1

# tokenizer.texts_to_sequences([all_captions[0]])[0]

max_caption_len = max([len(i.split()) for i in all_captions])

# Define encoder
model = VGG16()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

# Get image features


def get_image_feature(img_name):
    img_path = os.path.join('test_image/', img_name)
    # img = load_img(img_path,target_size=(224,224))
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    feature = fe.predict(img, verbose=0)
    return feature


# Load trained model
reconstructed_model = keras.models.load_model("my_model.keras")


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


def enter_image_for_caption_generate(image_name):
    image_path = os.path.join('test_image/', image_name)
    # image = Image.open(image_path)
    # image = Image.open(image_path)
    # captions = caption_dict[image_name]
    # print("Actual_Captions--->")
    # for i in captions:
    #    print(i)
    # print('-'*50)
    feature = get_image_feature(image_name)
    y_pred = prediction(reconstructed_model, feature,
                        tokenizer, max_caption_len)
    y_pred = y_pred.replace('start', '')
    y_pred = y_pred.replace('end', '')
    print('predicted-->')
    print(y_pred)
    print('-'*50)
    # plt.imshow(image)
    return y_pred

# predicted_text = enter_image_for_caption_generate('3413973568_6630e5cdac.jpg')


def speech(text):
    engine = pyttsx3.init()
    if not pyttsx3.init():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else :
        engine.say(text)
        engine.runAndWait()

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Theme: Eyes4Blind",
                       page_icon="👁️", layout="wide")
    # Set logo
    image = Image.open('logo.PNG')
    st.image(image, caption='Techwiz4: Aptech Can Tho')
    # Set page header
    st.header('Upload Image')
    st.title(":star2: Theme: Eyes4Blind :star2:")

    html_temp = """
    <div style="background-color: yellow; padding: 15px; border-radius: 10px">
        <h2 style="color: black; text-align: center; margin: 0;">Unleashing the Power of AIML for Intelligent Solutions</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Choose an image you want to know the description")

    if uploaded_img is not None:
        image_data = uploaded_img.getvalue()
        st.image(image_data, use_column_width=True)
    if st.button("Predict", key='predict_button'):
        if uploaded_img is not None:
            result = enter_image_for_caption_generate(uploaded_img.name)
            st.success('The text prediction is: {}'.format(result))
            st.session_state.result = result
        else:
            st.warning("Please upload an image first!")
    if st.button("Speech", key='speech_button'):
        if 'result' in st.session_state:
            text = st.session_state.result
            st.write("Speech output:" + text)
            speech(text)
        else:
            st.warning("Please click 'Predict' to generate a result first!")


if __name__ == '__main__':
    main()
