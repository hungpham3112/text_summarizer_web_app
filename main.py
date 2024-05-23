import streamlit as st
from transformers import pipeline

# Load the summarization pipeline with error handling
try:
    summarizer = pipeline("summarization")
except Exception as e:
    st.error(f"Error loading the summarization model: {e}")
    st.stop()

# Streamlit app title
st.title("Text Summarizer with Hugging Face Transformers")

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.header("Enter Text")
    user_input = st.text_area("Enter text to summarize", height=300)

with col2:
    st.header("Summary Output")
    if st.button("Summarize"):
        if user_input.strip() == "":
            st.warning("Please enter some text to summarize.")
        else:
            try:
                # Perform summarization with sampling enabled
                summary = summarizer(user_input, max_length=150, min_length=30, do_sample=True, top_k=50, top_p=0.95)
                summarized_text = summary[0]['summary_text']

                # Display the summary
                st.write(summarized_text)
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed with ❤️ using Streamlit and Hugging Face Transformers</p>
    </div>
    """,
    unsafe_allow_html=True
)
