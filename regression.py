import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from data.stock_data import pull_single_stock
import torch.nn.functional as F
import numpy as np


# Define the advanced regression model
class AdvancedRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(AdvancedRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


# Function to process data and setup training environment
def prepare_data_and_model(stock_symbol: str):
    df = pull_single_stock(stock_symbol)
    df.drop(["Date", "Ticker"], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = pd.DataFrame(IterativeImputer().fit_transform(df), columns=df.columns)

    X = StandardScaler().fit_transform(df.drop("Close", axis=1))
    y = df["Close"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=64,
        shuffle=True,
    )
    model = AdvancedRegressionModel(X_train.shape[1]).cuda()

    return (
        train_data,
        torch.FloatTensor(X_test).cuda(),
        torch.FloatTensor(y_test).cuda(),
        model,
    )


# Training the model
def train_model(train_loader, X_test, y_test, model, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:

    # Final validation
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_test), y_test.unsqueeze(1))


# Main execution
train_loader, X_test, y_test, model = prepare_data_and_model("TSLA")
train_model(train_loader, X_test, y_test, model)

# Prediction
model.eval()
with torch.no_grad():
    last_row_scaled = X_test[-1].unsqueeze(0)
    prediction = model(last_row_scaled).item()
