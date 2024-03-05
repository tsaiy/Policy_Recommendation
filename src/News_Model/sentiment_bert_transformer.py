import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

file_path = 'twitter_training.csv'
df = pd.read_csv(file_path)

def classify_sentiment(text):
   
    tokens = tokenizer.encode(text, return_tensors='pt')
    outputs = model(tokens)
    probs = softmax(outputs.logits, dim=1)
    _, predicted_class = torch.max(probs, dim=1)
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_labels[predicted_class.item()]
    return predicted_sentiment

df['Predicted_Sentiment'] = df['Text'].apply(classify_sentiment)
output_file_path = '/content/drive/MyDrive/Conservative-Official/News_model/newfile.xlsx'
df.to_excel(output_file_path, index=False)
