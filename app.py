import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your trained model
model = load_model('fake_news.h5')

# Initialize a tokenizer with the same configuration as used in training
tokenizer = Tokenizer(num_words=10000)  # Adjust num_words as needed

def predict_fake_news(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed
    prediction = model.predict(padded_sequences)
    return 'fake' if prediction[0][0] > 0.5 else 'real'

# Streamlit app
st.title('Fake News Detection')

text_input = st.text_area('Enter news text:', height=200)

if st.button('Submit'):
    if text_input:
        prediction = predict_fake_news(text_input)
        st.write(f'Prediction: {prediction}')
    else:
        st.write('Please enter some text.')
