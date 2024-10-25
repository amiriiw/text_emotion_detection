import torch
import psycopg2
from dataclasses import dataclass
from transformers import BertTokenizer, BertForSequenceClassification


@dataclass
class DBParams:
    dbname: str
    user: str
    password: str
    host: str
    port: str


class TextEmotionPostgreSQL:
    def __init__(self, model_path: str, db_params: DBParams):
        """Initializes the model for sequence classification, sets up a label mapping for emotions, and establishes a database connection."""
        self.model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_map = {0: 'fear', 1: 'joy', 2: 'sadness', 3: 'anger', 4: 'surprise', 5: 'neutral', 6: 'disgust'}
        self.conn = psycopg2.connect(**db_params.__dict__)
        self.cursor = self.conn.cursor()

    def predict_emotion(self, text: str) -> str:
        """Predicts the emotion of the input text, saves the result to the database, and returns the predicted emotion label."""
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

        self.cursor.execute("INSERT INTO emotions (text, emotion) VALUES (%s, %s)", (text, text_emotion))
        self.conn.commit()
        return text_emotion

    def setup_emotion_table(self) -> None:
        """Creates the emotions table in the database if it does not already exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                emotion VARCHAR(20) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def close_connection(self) -> None:
        """Closes the database cursor and connection."""
        self.cursor.close()
        self.conn.close()


if __name__ == "__main__":
    db_params = DBParams(
        dbname='textdb',
        user='test',
        password='1001',
        host='localhost',
        port='5432'
    )

    db_interface = TextEmotionPostgreSQL(model_path="./", db_params=db_params)
    db_interface.setup_emotion_table()
    user_text = input("Enter your text: ")
    emotion = db_interface.predict_emotion(user_text)
    print(f"Detected emotion: {emotion}")
    db_interface.close_connection()
