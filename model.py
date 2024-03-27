import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
data_path = 'raw_data.csv'
df = pd.read_csv(data_path)

# Drop non-numeric and irrelevant columns
df = df.drop(['Unnamed: 0', 'Date', 'Ticker'], axis=1)

# Replace inf/-inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with column means
df.fillna(df.mean(), inplace=True)

# Separate features and target
X = df.drop('Close', axis=1).values
y = df['Close'].values

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and the most recent data for prediction
X_train, X_most_recent = X_scaled[:-1], X_scaled[-1:]
y_train = y[:-1]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_most_recent = torch.FloatTensor(X_most_recent).unsqueeze(0)  # Add batch dimension
y_train = torch.FloatTensor(y_train)

# DataLoader for the training data
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# Regression Model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

input_size = X_train.shape[1]
model = RegressionModel(input_size).cuda()  # Move model to GPU

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 100
for epoch in range(epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Prediction for the most recent row
X_most_recent = X_most_recent.cuda()  # Move to GPU
prediction = model(X_most_recent)
print(f'Prediction for the most recent "Close" price: {prediction.item()}')

