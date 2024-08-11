"""-----------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about transformers library.
       this file is the detector side of this project.
---------------------------------------------------"""
import torch  # https://pytorch.org/docs/stable/index.html
import sqlite3  # https://docs.python.org/3/library/sqlite3.html
from transformers import BertTokenizer, BertForSequenceClassification  # https://transformer.readthedocs.io/en/latest/
"""----------------------------------------------------------------------------------------------------------------"""


class TextEmotionSQLite:
    def __init__(self, model_path, db_path="emotions.db"):
        self.model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.label_map = {0: 'fear', 1: 'joy', 2: 'sadness', 3: 'anger', 4: 'surprise', 5: 'neutral', 6: 'disgust'}

    def predict_emotion(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            text_emotion = self.label_map[predicted_class_id]
        self.cursor.execute("INSERT INTO emotions (text, emotion) VALUES (?, ?)", (text, text_emotion))
        self.conn.commit()
        return text_emotion

    def setup_emotion_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                emotion TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def close_connection(self):
        self.conn.close()


if __name__ == "__main__":
    db_interface = TextEmotionSQLite(model_path="./")
    db_interface.setup_emotion_table()
    user_text = input("enter your text: ")
    emotion = db_interface.predict_emotion(user_text)
    print(f"Detected emotion: {emotion}")
    db_interface.close_connection()
"""-----------------------------"""
