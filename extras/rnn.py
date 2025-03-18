import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a small vocabulary.
vocab = ["I", "love", "recurrent", "neural"]
vocab_size = len(vocab)

# A helper function to convert a word to a one-hot vector
def word_to_onehot(word):
    # Create a one-hot vector of size equal to vocab_size.
    # Here we simply assume that the index of the word in vocab is its encoding.
    onehot = torch.zeros(1, 1, vocab_size)  # shape: [batch_size, seq_len, vocab_size]
    if word in vocab:
        index = vocab.index(word)
    else:
        index = 0  # unknown words become index 0
    onehot[0, 0, index] = 1.0
    return onehot

# Define an RNN-based language model.
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=1):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        # Linear layer to map hidden state to vocabulary logits.
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        # out shape: [batch, seq_len, hidden_size]
        # We take the output at the last time step.
        last_output = out[:, -1, :]  # shape: [batch, hidden_size]
        logits = self.decoder(last_output)  # shape: [batch, vocab_size]
        probabilities = F.softmax(logits, dim=1)
        return probabilities, hidden

# Instantiate the RNN model.
# Here, we use the vocab_size as the input_size for simplicity (using one-hot encoded vectors).
model = MyRNN(input_size=vocab_size, hidden_size=3, vocab_size=vocab_size, num_layers=1)

# Initialize the hidden state.
hidden_state = torch.zeros(1, 1, 3)  # [num_layers, batch_size, hidden_size]

# A sample sentence
sentence = ["I", "love", "recurrent", "neural"]

# Process each word and predict the next one.
for word in sentence:
    # Convert the current word into a one-hot encoded tensor.
    input_tensor = word_to_onehot(word)
    probabilities, hidden_state = model(input_tensor, hidden_state)
    # Get the index of the predicted next word.
    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_word = vocab[predicted_index]
    
    print("Input word: ", word)
    print("Predicted next word: ", predicted_word)
    print("Output probabilities:", torch.round(probabilities * 1e4) / 1e4)
    print("")