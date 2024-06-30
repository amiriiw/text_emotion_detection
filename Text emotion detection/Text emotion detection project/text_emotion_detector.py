"""----------------------------------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about transformers library.
       this file is the detector side of this project.
----------------------------------------------------------------------------------------------------"""
# import what we need:
import sys  # https://docs.python.org/3/library/sys.html
import torch  # https://pytorch.org/docs/stable/index.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import matplotlib.pyplot as plt  # https://matplotlib.org/stable/index.html
from collections import Counter  # https://docs.python.org/3/library/collections.html
from transformers import BertTokenizer, BertForSequenceClassification  # https://transformer.readthedocs.io/en/latest/
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # https://matplotlib.org/stable/index.html
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPlainTextEdit, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox  # https://www.riverbankcomputing.com/static/Docs/PyQt5/
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class TransformersApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.text_entry = QPlainTextEdit()
        self.emotion_label = QLabel('Last Detected Emotion: None')
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.emotion_labels = ['fear', 'joy', 'sadness', 'anger', 'surprise', 'neutral', 'disgust']
        self.emotion_values = [0] * len(self.emotion_labels)
        self.bar_width = 0.3  # Fixed width for each bar
        self.x_positions = np.arange(len(self.emotion_labels))
        self.bars = None
        self.detected_emotions = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Emotion Detector')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #DDD6D6; color: black;")

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.text_entry.setPlaceholderText("Enter your text here")
        self.emotion_label.setStyleSheet("font-size: 16px;")

        select_folder_btn = QPushButton('Select Model Folder', self)
        select_folder_btn.clicked.connect(self.select_model_folder)

        analyze_btn = QPushButton('Analyze Texts', self)
        analyze_btn.clicked.connect(self.analyze_texts)

        layout = QVBoxLayout()
        layout.addWidget(self.text_entry)
        layout.addWidget(self.emotion_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(select_folder_btn)
        btn_layout.addWidget(analyze_btn)
        layout.addLayout(btn_layout)

        layout.addWidget(self.canvas)

        central_widget.setLayout(layout)

        self.bars = self.ax.bar(self.x_positions, self.emotion_values, width=self.bar_width, align='center',
                                color=['blue', 'green', 'red', 'purple', 'orange', 'gray', 'brown'])
        self.ax.set_xticks(self.x_positions)
        self.ax.set_xticklabels(self.emotion_labels)
        self.ax.set_xlabel('Emotions')
        self.ax.set_ylabel('Count')
        self.ax.set_title('Distribution of Emotions in User Texts')
        self.figure.tight_layout()
        self.canvas.draw()

    def select_model_folder(self):
        self.model_path = QFileDialog.getExistingDirectory(self, 'Select Model Folder')
        if not self.model_path:
            QMessageBox.critical(self, 'Error', 'Invalid model path!')
            return

    def analyze_texts(self):
        if not self.model_path:
            QMessageBox.warning(self, 'Warning', 'Please select a model folder first.')
            return

        model = BertForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        label_map = {0: 'fear', 1: 'joy', 2: 'sadness', 3: 'anger', 4: 'surprise', 5: 'neutral', 6: 'disgust'}
        emoji_map = {
            'fear': '😨',
            'joy': '😊',
            'sadness': '😢',
            'anger': '😠',
            'surprise': '😲',
            'neutral': '😐',
            'disgust': '🤢'
        }

        user_texts = self.text_entry.toPlainText().strip().split('\n')
        last_emotion = None
        for user_text in user_texts:
            if user_text.strip():
                predicted_emotion = self.predict_emotion(model, tokenizer, label_map, user_text)
                self.detected_emotions.append(predicted_emotion)
                last_emotion = predicted_emotion

        self.update_emotion_counts(self.detected_emotions)
        self.update_plot()

        if last_emotion:
            self.emotion_label.setText(f'Last Detected Emotion: {last_emotion} {emoji_map.get(last_emotion, "")}')

        self.text_entry.clear()

    @staticmethod
    def predict_emotion(model, tokenizer, label_map, text):
        inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = label_map[predicted_class]
        return predicted_label

    def update_emotion_counts(self, emotions_detected):
        emotion_counts = Counter(emotions_detected)
        for i, label in enumerate(self.emotion_labels):
            self.emotion_values[i] = emotion_counts.get(label, 0)

    def update_plot(self):
        self.ax.clear()
        self.bars = self.ax.bar(self.x_positions, self.emotion_values, width=self.bar_width, align='center',
                                color=['blue', 'green', 'red', 'purple', 'orange', 'gray', 'brown'])
        self.ax.set_xticks(self.x_positions)
        self.ax.set_xticklabels(self.emotion_labels)
        self.ax.set_xlabel('Emotions')
        self.ax.set_ylabel('Count')
        self.ax.set_title('Distribution of Emotions in User Texts')
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TransformersApp()
    window.show()
    sys.exit(app.exec_())
# ------------------------
