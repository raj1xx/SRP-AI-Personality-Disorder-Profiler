import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Define class labels (same as during training)
class_labels = ['Anxiety', 'Normal', 'Depression', 'Suicidal', 'Stress', 'Bipolar', 'Personality Disorder']


# Define the model architecture
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_hidden, max_len, num_lstm=1):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, num_hidden, bidirectional=True, num_layers=num_lstm)
        self.linear = nn.Linear(2 * num_hidden * max_len, len(class_labels))

    def forward(self, x):
        x = self.embd(x)
        x, (h_n, c_n) = self.lstm(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.linear(x)
        return x

# Load tokenizer
with open(r"D:\clg\SRP-MAIN\tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Model parameters (must match the saved model)
vocab_size = len(tokenizer.word_index) + 1
embedding_size = 64
num_hidden = 64
max_len = 6300

# Initialize model
model = Model(vocab_size, embedding_size, num_hidden, max_len)

# Load saved weights
model.load_state_dict(torch.load(r"D:\clg\SRP-MAIN\mental_health_model.pth", map_location=torch.device('cpu')))
model.eval()

print("Model loaded successfully!")

# Prediction function
def predict(text):
    tokenized = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokenized, maxlen=max_len)
    x = torch.from_numpy(padded).long()

    with torch.no_grad():
        output = model(x)
        predicted_class = output.argmax(dim=1).item()

    return class_labels[predicted_class]

# Example test sample
sample_text = "I feel like there's no way out and I don't see a future for myself anymore."
prediction = predict(sample_text)
print(f"Predicted Class: {prediction}")
