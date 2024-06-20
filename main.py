import streamlit as st

from model import load_model
from utils import split_into_chunks

# Streamlit app title and description
st.title("Text Summarizer with Hugging Face Transformers")

# Create two columns for input and output
col1, col2 = st.columns(2)

# Create text input area in the first column
with col1:
    st.header("Enter Text")
    user_input = st.text_area("Enter text to summarize", height=300)

# Load the model outside of the main function to leverage caching
summarizer = load_model()

if summarizer is None:
    st.error("Failed to load the summarization model. Please try again later.")
    st.stop()

# Summarize button and output in the second column
with col2:
    st.header("Summary Output")
    if st.button("Summarize"):
        if not user_input.strip():
            st.warning("Please enter some text to summarize.")
        else:
            # Start animation
            with st.spinner("Summarizing..."):
                try:
                    # Split the input text into chunks
                    chunks = split_into_chunks(user_input)

                    # Summarize each chunk
                    chunk_summaries = []
                    for chunk in chunks:
                        summary = summarizer(
                            chunk,
                            max_length=500,
                            min_length=30,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            early_stopping=True,
                        )
                        chunk_summaries.append(summary[0]["summary_text"])

                    # Combine chunk summaries into a final summary
                    final_summary = " ".join(chunk_summaries)

                    # Display the summary in a container
                    with st.container():
                        st.write("### Summarized Text")
                        st.write(final_summary)
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
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
    <p>Developed with ❤️  by hungpham3112</p>
    </div>
    """,
    unsafe_allow_html=True,
)
