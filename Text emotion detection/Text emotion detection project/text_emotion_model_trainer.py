"""-----------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about transformers library.
       this file is the file which create a model for detect text emotion.
-----------------------------------------------------------------------"""
import torch  # https://pytorch.org/docs/stable/index.html
import pandas as pd  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
from torch.utils.data import Dataset, DataLoader  # https://pytorch.org/docs/stable/data.html
from transformers import BertTokenizer, BertForSequenceClassification  # https://transformer.readthedocs.io/en/latest/
"""----------------------------------------------------------------------------------------------------------------"""


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'fear': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'neutral': 5, 'disgust': 6}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label_map.get(label, 0), dtype=torch.long)
        }


class TextEmotionDetector:
    def __init__(self, dataset_path, model_name="bert-base-uncased", num_labels=7, max_length=128, batch_size=8, lr=5e-5, num_epochs=3):
        self.dataset = pd.read_csv(dataset_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.dataset = CustomDataset(
            texts=self.dataset['Clean_Text'].values,
            labels=self.dataset['Emotion'].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in self.dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_model(self, save_directory="./"):
        self.model.save_pretrained(save_directory)


if __name__ == "__main__":
    detector = TextEmotionDetector("dataset.csv")
    detector.train()
    detector.save_model()
"""-------------------"""
