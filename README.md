# Text Emotion Detection Project

Welcome to the **Text Emotion Detection Project**! This project is designed to train a model for detecting emotions in text using the Transformers library and PyTorch, and then use that model to classify emotions in real-time and store the results in an SQLite database.

## Overview

This project consists of two main components:

1. **text_emotion_model_trainer.py**: This script is responsible for training a BERT-based model to classify text into one of seven emotions: fear, joy, sadness, anger, surprise, neutral, or disgust.
2. **text_emotion_detector.py**: This script uses the trained model to predict emotions from user input and stores the results in an SQLite database.

## Libraries Used

The following libraries are used in this project:

- **[torch](https://pytorch.org/docs/stable/index.html)**: PyTorch, an open-source machine learning library for Python, is used for model training and inference.
- **[pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)**: Pandas is used for handling the dataset and loading the data.
- **[transformers](https://huggingface.co/docs/transformers/index)**: The Transformers library by Hugging Face provides the pre-trained BERT model and tokenizer used in this project.
- **[sqlite3](https://docs.python.org/3/library/sqlite3.html)**: The SQLite3 module is used for creating and interacting with a local SQLite database to store detected emotions.

## Detailed Explanation

### `text_emotion_model_trainer.py`

This script is the backbone of the project, responsible for training the emotion detection model. The key components of the script are:

- **CustomDataset Class**: This class inherits from PyTorch's Dataset class and is used to preprocess the text data. It tokenizes the text, encodes the labels, and prepares the data for model training.
- **TextEmotionDetector Class**: This class sets up the BERT model and the training pipeline. It includes functions to load the dataset, initialize the model, and perform training.
- **train() Function**: This function iterates through the dataset, performs forward passes, computes loss, and updates the model weights.
- **save_model() Function**: After training, this function saves the trained model to a specified directory for later use.

### `text_emotion_detector.py`

This script uses the trained model to predict emotions from user input and stores the results in an SQLite database. The key components of the script are:

- **TextEmotionSQLite Class**: This class handles model loading, emotion prediction, and database interactions.
- **predict_emotion() Function**: This function tokenizes the input text, performs emotion prediction using the loaded model, and stores the result in the SQLite database.
- **setup_emotion_table() Function**: This function creates an SQLite table (if it doesn't already exist) to store text and the corresponding detected emotion.
- **close_connection() Function**: This function closes the connection to the SQLite database.

### How It Works

1. **Model Training**:
    - The `text_emotion_model_trainer.py` script reads a CSV file containing text data and corresponding emotion labels.
    - The text is tokenized using the BERT tokenizer, and the model is trained on this data.
    - The trained model is saved for later use.

2. **Emotion Detection**:
    - The `text_emotion_detector.py` script loads the trained model and sets up an SQLite database to store the results.
    - The user inputs text, which is tokenized and passed through the model.
    - The predicted emotion is stored in the database, and the user is informed of the detected emotion.

### Dataset

The dataset used for training the model can be accessed via this [Dataset](https://drive.google.com/drive/folders/1cN4jA771DYQvLEATApjSx3YdpPVgZiUx?usp=sharing)

## Installation and Setup

To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/amiriiw/text_emotion_detection
    cd text_emotion_detection
    ```

2. Install the required libraries:

    ```bash
    pip install torch pandas transformers
    ```

3. Prepare your dataset (a CSV file with columns 'Clean_Text' and 'Emotion').

4. Run the model training script:

    ```bash
    python text_emotion_model_trainer.py
    ```

5. Use the trained model for emotion detection:

    ```bash
    python text_emotion_detector.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
