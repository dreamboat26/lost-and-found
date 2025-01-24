import torch
import torch.nn as nn
import torch.nn.functional as F

# %% VanillaLSTM


class VanillaLSTM(nn.Module):
    def __init__(self):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=18, hidden_size=168,
                            batch_first=True, num_layers=1, bidirectional=False)
        self.fc2 = nn.Linear(168, 12)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc2(out)
        return out

# %% LSTMDENSE


class LSTMDENSE(nn.Module):
    def __init__(self, input_size=18, hidden_size=168, num_layers=1):
        super(LSTMDENSE, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, num_layers=num_layers, bidirectional=False)

        self.fc1 = nn.Linear(168, 64)
        self.fc2 = nn.Linear(64, 12)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out

# x = torch.randn(3, 158, 18)
# model = LSTMDENSE()
# out = model(x)

# %% LSTMAutoEncoder


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden, cell):
        output, _ = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_layers=5):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = LSTMDecoder(input_size, hidden_size, num_layers)
        self.out = VanillaLSTM()

    def forward(self, x):
        hidden, cell = self.encoder(x)
        output = self.decoder(x, hidden, cell)
        output = self.out(output)
        return output
