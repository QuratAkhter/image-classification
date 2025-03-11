import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import streamlit as st

model = load_model('C:/Users/asad/Desktop/Image Classification/Image_Classifier.keras')

data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
img_width=180
img_height=180
st.header('Image Classification Model')

image = st.text_input('Enter the Image Name','Apple.jpeg')

image_load = tf.keras.utils.load_img(image,target_size=(img_width,img_height))
image_load=tf.keras.utils.array_to_img(image_load)
image_load= tf.expand_dims(image_load,0)

pred =model.predict(image_load)
score = tf.nn.softmax(pred)

st.image(image)
st.write("The given image is {} with accuracy of {:.2f}".format(data_cat[np.argmax(score)],np.max(score)*1000))