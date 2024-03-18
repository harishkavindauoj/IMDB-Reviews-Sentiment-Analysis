import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import pickle
import torch.nn.functional as F

# Load and preprocess the dataset
df = pd.read_csv('IMDB_Dataset.csv')

# Tokenize the text
def tokenize_text(text):
    return word_tokenize(text)

df['tokenized_text'] = df['review'].apply(tokenize_text)

# Build vocabulary
all_words = [word for tokens in df['tokenized_text'] for word in tokens]
word_counts = Counter(all_words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Save word_to_idx mapping
with open("word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)

# Convert text to numerical format
def text_to_tensor(tokens):
    return torch.tensor([word_to_idx.get(word, len(word_to_idx)) for word in tokens], dtype=torch.long)

df['tensorized_text'] = df['tokenized_text'].apply(text_to_tensor)

# Pad sequences
max_length = 150
padded_texts = pad_sequence(df['tensorized_text'], batch_first=True, padding_value=0)
padded_texts = padded_texts[:, :max_length]  # Trim sequences to max_length

# Convert sentiment to numerical format
label_to_idx = {'negative': 0, 'positive': 1}
df['numerical_label'] = df['sentiment'].map(label_to_idx)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(padded_texts, df['numerical_label'], test_size=0.2, random_state=42)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_texts, torch.tensor(train_labels.tolist()))
val_dataset = TensorDataset(val_texts, torch.tensor(val_labels.tolist()))

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # Max pooling over time
        cat = self.dropout(torch.cat(pooled, dim=1))  # Concatenate and dropout
        output = self.fc(cat)  # Fully connected layer
        return output

# Initialize the model, optimizer, and loss function
vocab_size = len(word_to_idx)
embedding_dim = 100
num_filters = 100
filter_sizes = [3, 4, 5]
output_dim = 2  # 2 classes: negative, positive
dropout = 0.5

model = CNN(vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Print device being used
print(f"Using device: {device}")

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    # Save the necessary information along with the model state dictionary
    checkpoint = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'num_filters': num_filters,
        'filter_sizes': filter_sizes,
        'output_dim': output_dim,
        'dropout': dropout,
        'model_state_dict': model.state_dict()
    }
    torch.save(checkpoint, 'cnn_sentiment_analysis_model.pt')

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    correct_val_preds = 0
    total_val_preds = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct_val_preds += (predicted == labels).sum().item()
            total_val_preds += labels.size(0)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val Acc: {correct_val_preds / total_val_preds}')
