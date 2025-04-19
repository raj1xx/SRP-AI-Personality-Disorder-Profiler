# 🧠 Personality Disorder Detection from Text using NLP and AI Counseling

A web-based application that leverages **BERT-based text classification** and **AI-powered chatbot counseling** to detect and provide support for various personality disorders based on social media-like text inputs. This project uses **Flask**, **TensorFlow**, and **LangChain-Groq** (LLaMA 3 model) for seamless integration between mental health screening and AI-assisted conversation.

---

## 📌 Features

- 🧠 **Text Classification** using a fine-tuned BERT model to identify:
  - Normal
  - Suicidal
  - Borderline Personality Disorder (BPD)
  - Bipolar Disorder
  - Schizotypal Personality Disorder
- 🤖 **Counseling Assistant** with tailored responses using Groq's LLaMA 3.3 model.
- 📊 **Probability Scores** for prediction confidence.
- 🔁 **Progressive Conversation** with automated mental health insights after every few user inputs.
- 🔐 Session-based architecture for personalized interaction.

<center>![image](https://github.com/user-attachments/assets/bf03d194-f00d-4a6a-a8e0-5c3b929ddf2a)</center>


---

## 🧬 Project Structure

```
├── bert.py                # Flask app for prediction + Groq counseling
├── chat.py                # Separate chatbot backend using LangChain-Groq
├── FINAL_SRP.ipynb        # Main research and training notebook (model dev + analysis)
├── tokenizer/             # Directory for the tokenizer files
├── bert_classifier_weights.h5  # Fine-tuned BERT model weights
├── templates/
│   ├── index.html         # UI for classifier
│   └── chat.html          # UI for chatbot
└── requirements.txt       # Python package dependencies
```

---
<p align="center">
 ![image](https://github.com/user-attachments/assets/5a3a8d4f-8168-4b83-8593-e492aa7d754d)
</p>


## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/personality-disorder-detector.git
cd personality-disorder-detector
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Add Tokenizer and Weights
Place the tokenizer files in:
```
./tokenizer/content/tokenizer/
```
Ensure the model weights file `bert_classifier_weights.h5` is in the root directory.

### 4. Set API Key
Update `GROQ_API_KEY` in `bert.py` and `chat.py` with your valid Groq API key.

### 5. Run the Apps
Run the classifier and counseling system:
```bash
python bert.py
```

Run the progressive therapy chatbot (optional):
```bash
python chat.py
```

---

## 📈 Model Performance

- **"Normal" Class Accuracy:** 94%
- **"Suicidal" Detection Accuracy:** 91%
- **Average AUC:** 0.95+
- Confusion matrix and ROC curves show strong differentiation across disorders.

---

## 🛠 Technologies Used

- 🧠 **BERT (Hugging Face Transformers)**
- 🧪 **TensorFlow / Keras**
- 🌐 **Flask**
- 🗣 **LangChain with Groq (LLaMA 3.3 70B)**
- 📈 **Numpy, Matplotlib, Seaborn (analysis)**

---

## 🔒 Ethical Considerations

This project prioritizes **user privacy**, **data anonymization**, and **ethical AI use** for mental health. No real user data is stored. This is **not a replacement for professional diagnosis**.

---

## 📚 Authors

- Neeraj J V  
- Raj Mohan R  
- Ranjith S 

---
