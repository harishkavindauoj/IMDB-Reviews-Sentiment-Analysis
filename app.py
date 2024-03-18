import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from torch import nn
import torch.nn.functional as F


app = Flask(__name__, template_folder='templates')

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)  # Add channel dimension for conv2d
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # Convolutional and relu activation
        pooled = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conved]  # Max pooling over time
        cat = self.dropout(torch.cat(pooled, dim=1))  # Concatenate and dropout
        output = self.fc(cat)  # Fully connected layer
        return output

# Load the trained model
def load_model():
    print("Loading model...")
    try:
        # Load checkpoint
        checkpoint = torch.load('cnn_sentiment_analysis_model.pt', map_location=torch.device('cuda'))
        print("Checkpoint loaded successfully.")
        # Extract model architecture parameters from checkpoint
        vocab_size = checkpoint['vocab_size']
        embedding_dim = checkpoint['embedding_dim']
        num_filters = checkpoint['num_filters']
        filter_sizes = checkpoint['filter_sizes']
        output_dim = checkpoint['output_dim']
        dropout = checkpoint['dropout']
        # Initialize model with architecture parameters
        model = CNN(vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout)
        print("Model initialized successfully.")
        # Load model state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dictionary loaded successfully.")
        model.eval()
        print("Model set to evaluation mode.")
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Pre-trained model file 'cnn_sentiment_analysis_model.pt' not found.")

# Load word_to_idx mapping
def load_word_to_idx():
    print("Loading word_to_idx mapping...")
    with open("word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)
    print("word_to_idx mapping loaded successfully.")
    return word_to_idx

# Initialize model and word_to_idx mapping
model = load_model()
word_to_idx = load_word_to_idx()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        tokens = word_tokenize(text.lower())
        print(tokens)
        # Filter out-of-vocabulary words
        tokens = [word for word in tokens if word in word_to_idx]
        if not tokens:
            return render_template('result.html', text=text, sentiment="Neutral", error_message="Input text is too short. Please enter a longer text.")
        inputs = torch.tensor([word_to_idx[word] for word in tokens], dtype=torch.long)
        inputs = inputs.unsqueeze(0)
        print(inputs)
        
        try:
            outputs = model(inputs)
        except RuntimeError as e:
            error_message = "Input text size is too small. Please enter a longer text."
            return render_template('result.html', text=error_message, sentiment=" ")
        
        # Print the shape and size of the input tensor
        print("Input tensor shape:", inputs.shape)
        
        predicted_class = torch.argmax(outputs, dim=1).item()
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        return render_template('result.html', text=text, sentiment=sentiment, error_message=None)




if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
