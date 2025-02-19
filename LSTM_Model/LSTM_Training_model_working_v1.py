## design of data set, data is the collection of different sentences
## at the end appended  a full document-
## working with good accuracy  -- now train the model in different tranches and save the model with good data.
## how to make interactive

## whats  next - 2 side paths 1. continue with LSTM and more data. 
                    ##          2. Fine tune or RAG the promising bitnet.cpp
import re
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
input_file = "C:\\MachineLearning\\DataSets\\Autosar\\pdf_to_text_CryptoDriver\\refined.txt"
with open(input_file, "r", encoding="utf-8") as infile:
    fulltext = infile.read()
text1 = fulltext
print(fulltext)
# Sample text dataset (replace with your actual data)
text = [
    "Autosar is the standards followed by automotive oems tiers and software suppliers CDH is the module in the autosar crypto stack which interacts with CryIf and HSM via proxy.",
    "HSM is the hardware security module which runs in isolation from host cores.HSM receives the interrupt from the CDH via proxy and then calls the task comm in Nano kernel",
    "This Nano kernel is event triggered OS After the task comm the job is forwarded to execute commands function which handles the job and returns the job result",
]
print(len(text))
text.append(fulltext)
print(len(text))
# Basic word-level tokenization (remove punctuation)
def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())

tokenized_text = [tokenize(sentence) for sentence in text]

# Build vocabulary
word_freq = Counter(word for sentence in tokenized_text for word in sentence)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_freq.most_common())}  # +2 to reserve special tokens
vocab["<pad>"] = 0  # Padding token
vocab["<unk>"] = 1  # Unknown word token

# Reverse mapping for decoding
idx_to_word = {idx: word for word, idx in vocab.items()}

# Convert words to indices
encoded_text = [[vocab.get(word, vocab["<unk>"]) for word in sentence] for sentence in tokenized_text]

# Define input-output pairs
SEQ_LEN = 32  # Sentence length in tokens
X, Y = [], []

for sentence in encoded_text:
    if len(sentence) < SEQ_LEN + 1:
        continue
    for i in range(len(sentence) - SEQ_LEN):
        X.append(sentence[i:i+SEQ_LEN])
        Y.append(sentence[i+1:i+SEQ_LEN+1])  # Shift by one word

X, Y = np.array(X), np.array(Y)

# Create PyTorch Dataset
class WordDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = WordDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



import torch.nn as nn
import torch.optim as optim

# Define LSTM model
class LSTM_LLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        super(LSTM_LLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Padding token index is 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden_state=None):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        output = self.fc(lstm_out)  # (batch, seq_len, vocab_size)
        return output, hidden_state

# Define model parameters
VOCAB_SIZE = len(vocab)
model = LSTM_LLM(VOCAB_SIZE)

# Print total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2 = optim.Adam(model.parameters(), lr=0.0002)
# Training loop

import os

checkpoint_path = "checkpoint.pth"
start_epoch = 0

# Optionally, load an existing checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Assuming you're using optimizer or optimizer2 based on the epoch:
    # For simplicity, we'll load the optimizer state for the initial optimizer.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")

# Training loop
num_epochs = 40
for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(x_batch)  # (batch, seq_len, vocab_size)
        loss = criterion(outputs.view(-1, VOCAB_SIZE), y_batch.view(-1))
        loss.backward()

        # Use different optimizers based on epoch if necessary:
        if epoch < 15:
            optimizer.step()
        else:
            optimizer2.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # Save a checkpoint at the end of each epoch
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()  # Or use optimizer2 if appropriate
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

# Finally, you can also save the final model state separately:
torch.save(model.state_dict(), "lstm_model_final.pth")


#num_epochs = 40
#for epoch in range(num_epochs):
#    total_loss = 0
#    for x_batch, y_batch in dataloader:
#        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

#        optimizer.zero_grad()
#        outputs, _ = model(x_batch)  # (batch, seq_len, vocab_size)

        # Reshape for loss computation
#        loss = criterion(outputs.view(-1, VOCAB_SIZE), y_batch.view(-1))
#        loss.backward()
#        if(epoch < 15):
#            optimizer.step()
#        else:
#            optimizer2.step()

#        total_loss += loss.item()

#    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
# Save the model's state dictionary
#torch.save(model.state_dict(), "lstm_model_1.pth")

import torch.nn.functional as F

model = LSTM_LLM(VOCAB_SIZE, embedding_dim=256, hidden_dim=512, num_layers=2)

# Load the saved state dictionary
model.load_state_dict(torch.load("lstm_model_final.pth"))

# Set the model to evaluation mode (if you are going to use it for inference)
model.eval()

def generate_text_top_k(model, seed_text, max_len=20, k=5):
    model.eval()
    words = tokenize(seed_text)
    print(seed_text.lower())
    input_seq = torch.tensor([[vocab.get(word, vocab["<unk>"]) for word in words]], dtype=torch.long).to(device)
    words1 = []
    hidden_state = None
    with torch.no_grad():
        for _ in range(max_len):
            output, hidden_state = model(input_seq, hidden_state)
            logits = output[:, -1, :]  # Last token predictions
            probs = F.softmax(logits, dim=-1)

            # Sample from the top-k most likely words
            top_k_probs, top_k_indices = torch.topk(probs, k)
            top_k_probs = top_k_probs.squeeze()
            top_k_indices = top_k_indices.squeeze()

            next_word_idx = top_k_indices[torch.multinomial(top_k_probs, 1).item()].item()
            next_word = idx_to_word.get(next_word_idx, "<unk>")

            words.append(next_word)
            words1.append(next_word)
            input_seq = torch.tensor([[next_word_idx]], dtype=torch.long).to(device)

    return " ".join(words1)

# Try generating again
print(generate_text_top_k(model, "AES is the", max_len=20, k=5))


