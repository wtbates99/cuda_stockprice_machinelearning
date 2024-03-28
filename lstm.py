import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from data.stock_data import pull_single_stock

# Load the dataset
df = pull_single_stock("TSLA")

# Preprocessing
df = df.drop(['Date', 'Ticker'], axis=1)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = IterativeImputer()
df_imputed = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed, columns=df.columns)
X = df.drop('Close', axis=1).values
y = df['Close'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating sequences
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

seq_length = 5
X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

# Convert to PyTorch tensors
X_seq = torch.FloatTensor(X_seq)
y_seq = torch.FloatTensor(y_seq)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
input_size = X_seq.shape[2]
hidden_size = 50
num_layers = 4
model = LSTMModel(input_size, hidden_size, num_layers)
model.cuda()  # Use CUDA

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_data_seq = TensorDataset(X_seq, y_seq)
train_loader_seq = DataLoader(dataset=train_data_seq, batch_size=64, shuffle=True)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader_seq:
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# Predicting tomorrow's Close price
model.eval()
with torch.no_grad():
    last_seq = X_scaled[-seq_length:].reshape(1, seq_length, input_size)
    last_seq = torch.FloatTensor(last_seq).cuda()
    predicted_close = model(last_seq).item()

print(f"Predicted tomorrow's 'Close' price: {predicted_close}")

