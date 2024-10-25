# Text Emotion Detection using BERT

This repository provides two main scripts for detecting emotions in text using a BERT model: `train.py` for training the model, and `detect.py` for predicting emotions from new text inputs. The model classifies text into seven different emotions: fear, joy, sadness, anger, surprise, neutral, and disgust.

## Features
- **Emotion Classification**: Classifies text into seven emotions using a pre-trained BERT model.
- **Custom Dataset Handling**: Loads a dataset for training and processes it into a format suitable for BERT.
- **Database Integration**: Saves detected emotions into a PostgreSQL database for record-keeping.

## Libraries Used
This project uses the following libraries:
- [PyTorch](https://pytorch.org/docs/stable/index.html) for model implementation.
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index) for pre-trained BERT model and tokenizer.
- [Pandas](https://pandas.pydata.org/docs/) for data handling and preprocessing.
- [Psycopg2](https://www.psycopg.org/docs/) for PostgreSQL database interaction.

## Introduction

### File: `train.py`
This script is responsible for training a BERT model on an emotion classification dataset. The main class and functions include:

#### Classes and Functions

- **Class `CustomDataset`**: 
  A PyTorch Dataset for managing text data and corresponding emotion labels.
  
  - `__init__(self, texts, labels, tokenizer, max_length)`: Initializes the dataset with text data, labels, a tokenizer, and the maximum sequence length.
  - `__len__(self)`: Returns the number of samples in the dataset.
  - `__getitem__(self, idx)`: Retrieves the text and label at a specific index and tokenizes the text.

- **Class `TextEmotionDetector`**:
  Manages the model setup, dataset loading, and training loop for BERT.
  
  - `__init__(self, dataset_path, model_name, num_labels, max_length, batch_size, lr, num_epochs, device)`: Loads the dataset, sets up the tokenizer and model, and defines the training parameters.
  - `train(self)`: Trains the model over a specified number of epochs, logging the average loss per epoch.
  - `save_model(self, save_directory)`: Saves the trained model and tokenizer to the specified directory.

### File: `detect.py`
This script is used for detecting emotions from new text inputs and saving the results in a PostgreSQL database. The main classes and functions are:

#### Classes and Functions

- **Class `DBParams`**: 
  A data structure for managing PostgreSQL database connection parameters.
  
- **Class `TextEmotionPostgreSQL`**:
  Loads a pre-trained model, establishes a database connection, and provides emotion detection functionality.
  
  - `__init__(self, model_path, db_params)`: Initializes the model for text classification, sets up emotion label mappings, and connects to the PostgreSQL database.
  - `predict_emotion(self, text)`: Predicts the emotion of the input text, saves the result in the database, and returns the predicted emotion label.
  - `setup_emotion_table(self)`: Creates a table in the database to store text and detected emotions if the table does not already exist.
  - `close_connection(self)`: Closes the database connection.

## Usage

### Training the Model

1. Prepare a dataset in CSV format with two columns: `Clean_Text` and `Emotion`, and place it in the same directory as `train.py`.
2. Adjust the `dataset_path` in `train.py` to point to your CSV file.
3. Run the training script:
   ```bash
   python3 train.py
   ```

   This will train the model and save it to the specified directory (default is `./`).

### Detecting Emotion

1. Ensure PostgreSQL is set up and the connection parameters are configured correctly in `detect.py`.
2. Run the detection script:
   ```bash
   python3 detect.py
   ```

3. Enter the text to analyze when prompted. The detected emotion will be printed in the console and saved in the database.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/amiriiw/text_emotion_detection
   cd text-emotion-detection
   cd Text-emotion-detection
    ```

2. Install the required packages:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Make sure PostgreSQL is installed and running, and create a database to use with this project.

4. Download the dataset via this link: [Drive](https://drive.google.com/drive/folders/1aMBDv4xQ0d9LOjX-3xoeodY49MjcPkKA?usp=sharing)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
