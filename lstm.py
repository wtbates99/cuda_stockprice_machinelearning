import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from data.stock_data import pull_single_stock

def preprocess_data(stock_symbol: str) -> pd.DataFrame:
    """Load and preprocess stock data."""
    df = pull_single_stock(stock_symbol)
    df.drop(['Date', 'Ticker'], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = IterativeImputer()
    df_imputed = imputer.fit_transform(df)
    return pd.DataFrame(df_imputed, columns=df.columns)

def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features using standard scaling."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> tuple:
    """Create sequences from the data for time series forecasting."""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def train_model(model: nn.Module, train_loader: DataLoader, criterion, optimizer, epochs: int = 100) -> None:
    """Train the LSTM model."""
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

def predict_close_price(model: nn.Module, last_seq: np.ndarray) -> float:
    """Predict the next closing price."""
    model.eval()
    with torch.no_grad():
        last_seq = torch.FloatTensor(last_seq).cuda()
        predicted_close = model(last_seq).item()
    return predicted_close

# Main execution
df = preprocess_data("TSLA")
X = df.drop('Close', axis=1).values
y = df['Close'].values
X_scaled = normalize_features(X)
seq_length = 5
X_seq, y_seq = create_sequences(X_scaled, y, seq_length)
X_seq, y_seq = torch.FloatTensor(X_seq).cuda(), torch.FloatTensor(y_seq).cuda()

input_size = X_seq.shape[2]
model = LSTMModel(input_size, hidden_size=50, num_layers=4).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=64, shuffle=True)

train_model(model, train_loader, criterion, optimizer)

last_seq = X_scaled[-seq_length:].reshape(1, seq_length, input_size)
predicted_close = predict_close_price(model, last_seq)
print(f"Predicted tomorrow's 'Close' price: {predicted_close}")
