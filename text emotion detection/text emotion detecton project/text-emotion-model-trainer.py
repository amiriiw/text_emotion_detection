"""----------------------------------------------------------------------------------------------------
well come, this is maskiiw, this is a simple project about transformers library.
       this file is the file which create a model.
----------------------------------------------------------------------------------------------------"""
# import what we need:
import torch  # https://pytorch.org/docs/stable/index.html
import pandas as pd  # # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
from torch.utils.data import Dataset, DataLoader  # https://pytorch.org/docs/stable/data.html
from transformers import BertTokenizer, BertForSequenceClassification  # https://transformer.readthedocs.io/en/latest/
# --------------------------------------------------------------------------------------------------------------------


class Transformers:

    df = pd.read_csv('emotion_dataset.csv')

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
                return_token_type_ids=False,
                truncation=True
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.label_map.get(label, 0), dtype=torch.long)
            }
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)

    dataset = CustomDataset(
        texts=df['Clean_Text'].values,
        labels=df['Emotion'].values,
        tokenizer=tokenizer,
        max_length=128
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    num_epochs = 3
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.save_pretrained("./")
# ------------------------------
