import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
irom sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from data.stock_data import pull_single_stock


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def preprocess_and_transform_data(stock_symbol: str, seq_length: int) -> tuple:
    df = pull_single_stock(stock_symbol)
    df.drop(["Date", "Ticker"], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputed_data = IterativeImputer().fit_transform(df)
    X = StandardScaler().fit_transform(imputed_data)
    y = df["Close"].values

    X_seq = np.array([X[i : (i + seq_length)] for i in range(len(X) - seq_length)])
    y_seq = np.array([y[i + seq_length] for i in range(len(X) - seq_length)])
    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)


# Main execution
seq_length = 5
X_seq, y_seq = preprocess_and_transform_data("TSLA", seq_length)
model = LSTMModel(X_seq.shape[2], hidden_size=50, num_layers=4).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(
    TensorDataset(X_seq.cuda(), y_seq.cuda()), batch_size=64, shuffle=True
)

# Train model
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()

# Predict next close
last_seq = X_seq[-1].unsqueeze(0).cuda()
predicted_close = model(last_seq).item()
