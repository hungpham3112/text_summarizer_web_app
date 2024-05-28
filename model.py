import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
        return summarizer
    except Exception as e:
        st.error(f"Error loading the summarization model: {e}")
        return None
