import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

st.title("Emotion Detection Demo")

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "./emotion_model",
        num_labels=28,
        problem_type="multi_label_classification"
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

emotion_labels = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

user_input = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].numpy()

    predictions = [
        emotion_labels[i] for i, p in enumerate(probs) if p > 0.5
    ]

    if not predictions:
        top = np.argmax(probs)
        predictions = [emotion_labels[top]]

    st.write("Predicted emotions:", predictions)
    st.write("Confidence scores:")
    st.write({emotion_labels[i]: float(probs[i]) for i in range(len(probs))})
