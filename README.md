# Text Summarizer App 📄🔍

## Description
The Text Summarizer App is a powerful tool designed to generate concise summaries of long texts. It leverages the `bart-large-cnn` model, a transformer-based model, to extract key information and present it in a clear, succinct manner. Ideal for users who need to quickly understand the main points of lengthy documents, articles, or reports.

## Installation Instructions

### Running Locally 🖥️

1. **Clone the Repository**
    ```bash
    git clone https://github.com/hungpham3112/text_summarizer_web_app.git
    cd text_summarizer_web_app
    ```

2. **Create a Conda Environment**
    ```bash
    conda create --name text-summarizer python=3.10
    conda activate text-summarizer
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**
    ```bash
    streamlit run main.py
    ```

### Running on Cloud ☁️

- The app has been deployed on Streamlit cloud, so you can following this link to use:
[https://textsummarizerr.streamlit.app/](https://textsummarizerr.streamlit.app/)

## Usage Instructions 📽️

To use the Text Summarizer App, simply follow the instructions provided in the video below:

![Usage Instructions](demo.gif)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
