import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import keras
from PIL import Image
import streamlit as st
from process_functions import readImage, process, idx_to_word, prediction, speech_audio
# Read caption
img_id_caption = pd.read_csv("flickr8k/captions.txt", sep=',')
print(len(img_id_caption))

img_id_caption['cleaned_caption'] = img_id_caption['caption'].apply(
    lambda x: 'start '+' '.join(process(x)) + ' end')
all_captions = img_id_caption['cleaned_caption'].to_list()
print(len(all_captions))

img_id_caption.drop(columns=['caption'], inplace=True)
# Convert DataFrame to dictionary with lists for duplicate keys
caption_dict = {k: v.tolist() for k, v in img_id_caption.groupby('image')['cleaned_caption']}

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
reconstructed_model = keras.models.load_model("models/my_model.keras")

# Prediction caption - add 'start' and 'end' to caption
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

def main():
    # Set page title and favicon
    st.set_page_config(page_title="Theme: Eyes4Blind",
                       page_icon="👁️", layout="wide")
    # Set logo
    logo = Image.open('Logo.png')

    st.image(logo, use_column_width=True)

    # Set captions with different elements
    st.markdown("<caption style='text-align: left; float: left; width: 50%;'> Techwiz4: Aptech Can Tho </caption>", unsafe_allow_html=True)

    st.markdown("<caption style='text-align: right; float: right; width: 50%;'> Unleashing the Power of AIML for Intelligent Solutions </caption>", unsafe_allow_html=True)

    # Set page header
    st.markdown("<h1 style='text-align: center;'> Theme: Eyes4Blind </h1>", unsafe_allow_html=True)

    # Set PyplotGlobalUseWarning False
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Upload image
    uploaded_img = st.file_uploader(
        "Choose an image you want to know the description")

    if uploaded_img is not None:
        # Read the uploaded image using OpenCV
        image_data = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), 1)

        # Display the image using Matplotlib
        plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        st.pyplot()

        # Get the name of the uploaded image
        image_name = uploaded_img.name

        # Create 'test_image' folder if it doesn't exist
        os.makedirs("test_image", exist_ok=True)

        # Save the image to the 'test_image' folder using OpenCV
        save_path = os.path.join("test_image", image_name)
        cv2.imwrite(save_path, image_data)

        # Button Prediction caption from image uploaded
    if st.button("Predict", key='predict_button'):
        if uploaded_img is not None:
            result = enter_image_for_caption_generate(uploaded_img.name)
            st.success('The text prediction is: {}'.format(result))
            speech_audio(result)
            st.session_state.result = result
        else:
            st.warning("Please upload an image first!")

if __name__ == '__main__':
    main()
