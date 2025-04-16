import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 1: Create a Simple Dataset
sentences = ["I love programming", "You love programming", "Python is fun", "Deep learning is amazing", "I enjoy coding", "The sky is blue", "Artificial intelligence is the future"]

# Tokenize the sentences
tokenized_sentences = [sentence.split() for sentence in sentences]

# Create a vocabulary 
tokenized_sentences = [sentence.split() for sentence in sentences]

# Create a vocabulary 
vocab = set(word for sentence in tokenized_sentences for word in sentence)
vocab = list(vocab)
vocab_size = len(vocab)

# Create word-to-index and index-to-word mappings
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Step 2: Prepare the Data
sequences = [[word_to_idx[word] for word in sentence] for sentence in tokenized_sentences]

# Create input-output pairs
input_sequences = []
output_sequences = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i])
        output_sequences.append(sequence[i])

# Pad sequences to the same length
max_len = max(len(seq) for seq in input_sequences)
input_sequences = [seq + [0] * (max_len - len(seq)) for seq in input_sequences] # Padding with 0

# Convert to PyTorch tensors
input_sequences = torch.tensor(input_sequences, dtype=torch.long)
output_sequences = torch.tensor(output_sequences, dtype=torch.long)

# Step 3: Define the Model
class NextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NextTokenPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim * max_len, vocab_size)
    
    def forward(self,x):
        embedded = self.embedding(x)
        embedded = embedded.view(embedded.size(0), -1) # Flatten the embeddings
        output = self.fc(embedded)
        return output

# Hyperparameters
embedding_dim = 10
hidden_dim = 16
learning_rate = 0.01
epochs = 100

# Initialize the model, loss function and optimizer
model = NextTokenPredictor(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 4: Train the Model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_sequences)
    loss = criterion(outputs, output_sequences)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1} / {epochs}], Loss: {loss.item():.4f}")

# Step 5: Test the Model
def predict_next_token(model, input_sequence, word_to_idx, idx_to_word):
    model.eval()
    with torch.no_grad():
        # Convert input sequences to tensor and add batch dimension (dim=0)
        input_indices = [word_to_idx[word] for word in input_sequence]
        #input_sequences = torch.tensor([word_to_idx[word] for word in input_sequence], dtype=torch.long).unsqueeze(0)
        # Pad the input tensor to match max_len
        padded_input = input_indices + [0] * (max_len - len(input_indices))
        #padded_input = torch.cat([input_sequences, torch.zeros(1, max_len - input_sequences.size(1), dtype=torch.long)], dim=1)
        # Convert the tensor and add batch dimension
        input_tensor = torch.tensor(padded_input, dtype = torch.long).unsqueeze(0)
        # Get Predictions
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_token_idx = torch.multinomial(probabilities, 1).item()
        return idx_to_word[predicted_token_idx]

# Example prediction
input_sequence = ["Artificial", "intelligence"]
predicted_token = predict_next_token(model, input_sequence, word_to_idx, idx_to_word)
print(f"Input: {' '.join(input_sequence)}")
print(f"Predicted next token: {predicted_token}")


