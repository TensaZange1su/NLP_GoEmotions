# Emotion Detection from Text using Transformers, Embeddings, and LLM Prompting

Author: Almas  
Course: Natural Language Processing  
Approach: A (Custom Model + Comparative Study)

---

# Project Overview

This project builds and compares multiple NLP approaches for multi-label emotion detection using the GoEmotions dataset.

The following modeling paradigms are implemented:

1. Fine-tuned DistilBERT
2. RoBERTa classifier
3. Pretrained SentenceTransformer embeddings + Logistic Regression
4. LLM Zero-Shot prompting
5. LLM Few-Shot prompting

The goal is to compare supervised fine-tuning, embedding-based models, and prompting-based methods.

---

# Dataset

Dataset: GoEmotions (Google Research)  
Source: HuggingFace  
Size:

- Train: 43,410
- Validation: 5,426
- Test: 5,427

Labels:

27 emotions + neutral

Task type: Multi-label classification

Example:

Text:
"I feel exhausted and overwhelmed."

Labels:
sadness, fatigue

---

# Models Implemented

## 1. Fine-tuned DistilBERT

Architecture:

Text  
→ Tokenizer  
→ DistilBERT encoder (6 layers)  
→ Linear classifier  
→ Sigmoid output  

Loss: BCEWithLogitsLoss  
Optimizer: AdamW  
Epochs: 5  

Advantages:

- Lightweight
- Fast training
- High accuracy

---

## 2. RoBERTa classifier

Architecture:

Text  
→ Tokenizer  
→ RoBERTa encoder  
→ Linear classifier  
→ Sigmoid  

Used as strong transformer baseline.

---

## 3. SentenceTransformer Embeddings

Model:

all-MiniLM-L6-v2

Pipeline:

Text  
→ SentenceTransformer  
→ 384-d embedding  
→ Logistic Regression (One-vs-Rest)

Advantages:

- Very fast
- Lightweight

Disadvantages:

- Lower accuracy than fine-tuned transformers

---

## 4. LLM Zero-Shot

Uses GPT API with prompt:

"Classify emotions from this list..."

No training required.

---

## 5. LLM Few-Shot

Uses GPT API with labeled examples in prompt.

Improves performance over zero-shot.

---

# Evaluation Metrics

The following metrics are used:

- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Accuracy
- Confusion matrix

---

# Results Summary

Based on experiments:

Best performance:
Fine-tuned DistilBERT

Second best:
RoBERTa

Moderate performance:
Embedding model

Lowest performance:
LLM Zero-shot

LLM Few-shot improves over zero-shot but remains weaker than fine-tuned models.

---

# Project Structure

project/
│
├── app.py
├── requirements.txt
├── README.md
├── emotion_model/
│ └── checkpoint-13570/
│ ├── config.json
│ ├── model.safetensors
│ └── tokenizer files
│
├── notebooks/
│ └── final_nlp_almas.ipynb

---

# Installation

Install dependencies:

pip install -r requirements.txt

---

# Run Streamlit App

streamlit run app.py

---

# Example Output

Input:

"I feel exhausted and overwhelmed."

Output:

sadness  
fatigue  

with confidence scores.

---

# Training

DistilBERT was fine-tuned using:

Epochs: 5  
Batch size: 16  
Learning rate: 2e-5  

Loss function:

BCEWithLogitsLoss

---

# Ethical Considerations

Important risks:

Dataset bias (Reddit-based)

Cultural bias

Mental health sensitivity

Model errors in ambiguous cases

Human oversight recommended in real applications

---

# Future Improvements

Class balancing  
Data augmentation  
LoRA fine-tuning  
Multilingual extension  

---

# License

Educational use only.
