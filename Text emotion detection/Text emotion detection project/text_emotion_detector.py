"""----------------------------------------------------------------------------------------------------
well come, this is maskiiw, this is a simple project about transformers library.
       this file is the detector side of this project.
----------------------------------------------------------------------------------------------------"""
# import what we need:
import torch  # https://pytorch.org/docs/stable/index.html
import matplotlib.pyplot as plt  # https://matplotlib.org/stable/index.html
from collections import Counter  # https://docs.python.org/3/library/collections.html
from transformers import BertTokenizer, BertForSequenceClassification  # https://transformer.readthedocs.io/en/latest/
# --------------------------------------------------------------------------------------------------------------------


class transformers:

    model = BertForSequenceClassification.from_pretrained("./", local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    label_map = {0: 'fear', 1: 'joy', 2: 'sadness', 3: 'anger', 4: 'surprise', 5: 'neutral', 6: 'disgust'}

    @staticmethod
    def predict_emotion(text):
        inputs = (transformers.tokenizer
                  (text, return_tensors="pt", max_length=128, padding="max_length", truncation=True))
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            outputs = transformers.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = transformers.label_map[predicted_class]
        return predicted_label


emotions_detected = []
while True:
    user_text = input("Enter a text to detect emotion:  (write 'done' to exit)")
    if user_text == 'done':
        break
    predicted_emotion = transformers.predict_emotion(user_text)
    print("Predicted Emotion:", predicted_emotion)
    emotions_detected.append(predicted_emotion)

emotion_counts = Counter(emotions_detected)
emotions_labels = list(emotion_counts.keys())
emotions_values = list(emotion_counts.values())

plt.bar(emotions_labels, emotions_values)
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.title('Distribution of Emotions in User Texts')
plt.show()
# --------
