import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import numpy as np

# Загрузка предобученной модели RuBERT
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

# Загрузка предобученной модели логистической регрессии
lr_model = joblib.load('trained_logistic_regression_model.pkl') 

class_mapping = {
    1: 'Fashion',
    2: 'Sport',
    3: 'Technologies',
    4: 'Finances',
    0: 'Cryptocurrency'
}

# Функция для классификации текста
def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings

st.title('Determine the topic of the news 📑')
user_input = st.text_area('Enter text for classification:', '')

if st.button('Classify🔥'):
    if user_input.strip() == '':
        st.error('Please enter text for classificationу!😡')
    else:
        X= embed_bert_cls(user_input, model, tokenizer)
        predictions= lr_model.predict(X)
        res=class_mapping[predictions[0]]
        st.success(f'Predicted text topic: {res}')



