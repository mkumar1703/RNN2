#Step 1: Import Libraries and load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}  
# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_model.h5')
#model = load_model("simple_rnn_model.h5", custom_objects={})

#Step 2: Helper Functions
# Function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

import streamlit as st

# Page Configuration
st.set_page_config(page_title="🎬 IMDB Sentiment Analyzer", layout="centered")

# Custom CSS for styling with font color fix
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea, .stButton button {
        font-size: 16px !important;
    }
    .stButton button {
        background-color: #6c63ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 2em;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #5848c2;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffffff;
        color: #333333; /* <-- FIX: font color set to dark gray */
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("## 🎥 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review and let our AI model predict its **sentiment**! 🎭")

# Text Input
user_input = st.text_area("📝 Movie Review", height=150, placeholder="Type your review here...")

# Classify Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() != "":
        # Example placeholder logic
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = "👍 Positive" if prediction[0][0] > 0.5 else "👎 Negative"
        score = prediction[0][0]

        # Display Result in styled box
        st.markdown(f"""<div class='result-box'>
            <h4>🎯 Prediction Result</h4>
            <p><strong>Sentiment:</strong> {sentiment}</p>
            <p><strong>Confidence Score:</strong> {score:.2f}</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Please enter a movie review to analyze.")
else:
    st.info("Awaiting input for sentiment analysis...")
