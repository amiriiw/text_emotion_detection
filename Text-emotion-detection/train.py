import torch
import logging
import pandas as pd
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: BertTokenizer, max_length: int):
        """Initializes a dataset object with texts, labels, tokenizer, and maximum sequence length. Sets up a label mapping dictionary for emotion categories."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'fear': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'neutral': 5, 'disgust': 6}

    def __len__(self):
        """Returns the number of texts in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int):
        """Retrieves the text, label, and tokenized encoding for a given index in the dataset. Returns a dictionary containing input IDs, attention mask, and the corresponding label."""
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.label_map.get(label, 0), dtype=torch.long)
        }


class TextEmotionDetector:
    def __init__(self, dataset_path: str, model_name: str = "bert-base-uncased", num_labels: int = 7, max_length: int = 128, batch_size: int = 8, lr: float = 5e-5, num_epochs: int = 3, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initializes the model training setup, including loading the dataset, tokenizer, model, and training parameters. Sets up the data loader and optimizer."""
        logger.info(f"Loading dataset from {dataset_path}")
        self.dataset_df = pd.read_csv(dataset_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

        self.dataset = CustomDataset(
            texts=self.dataset_df['text'].values,
            labels=self.dataset_df['emotion'].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self):
        """Trains the BERT model for a specified number of epochs using the provided data loader. Logs the progress and computes the average loss per epoch."""
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} started.")

            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
            torch.cuda.empty_cache()

    def save_model(self, save_directory: str = "./"):
        """Saves the trained model and tokenizer to the specified directory."""
        logger.info(f"Saving model to {save_directory}")
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)


if __name__ == "__main__":
    detector = TextEmotionDetector("dataset.csv")
    detector.train()
    detector.save_model()
