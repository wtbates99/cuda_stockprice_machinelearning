import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Replace inf/-inf with NaN and fill NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# Assuming the 'df' DataFrame is sorted in ascending order by date
num_days_backtest = 90

# Split the dataset into features and target without the last row (reserved for prediction)
X = df.drop(['Close'], axis=1).values[:-1]
y = df['Close'].values[:-1]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolate the last 90 days for backtesting
if len(X_scaled) > num_days_backtest:
    X_backtest = X_scaled[-num_days_backtest:]
    y_backtest = y[-num_days_backtest:]
    # The rest is for training
    X_train_scaled = X_scaled[:-num_days_backtest]
    y_train = y[:-num_days_backtest]
else:
    # Fallback in case there's not enough data; adjust as needed
    X_backtest = X_scaled
    y_backtest = y
    X_train_scaled, y_train = np.array([]), np.array([])  # Example fallback, adjust based on your needs

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_backtest_tensor = torch.FloatTensor(X_backtest)
y_backtest_tensor = torch.FloatTensor(y_backtest)

# Prepare the most recent data point
X_most_recent_raw = df.drop(['Close'], axis=1).iloc[-1, :].values.reshape(1, -1)
X_most_recent_scaled = scaler.transform(X_most_recent_raw)  # Use the same scaler as for the training data
X_most_recent = torch.FloatTensor(X_most_recent_scaled)

# Dataloaders for the training and testing data
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_backtest_tensor, y_test_backtest_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

# Enhanced Regression Model with additional layers and ReLU activation
class EnhancedRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(EnhancedRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

input_size = X_train.shape[1]
model = EnhancedRegressionModel(input_size).cuda()

# Loss, Optimizer, and Scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training Loop with CUDA utilization and early stopping
epochs = 100
patience = 10
min_val_loss = np.inf
patience_counter = 0

for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Validation loop
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            val_loss += criterion(outputs, targets.unsqueeze(1)).item()
    
    val_loss /= len(test_loader)
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')
        # Early stopping logic
        
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    scheduler.step()

import math
from sklearn.metrics import mean_squared_error

# Function to calculate RMSE
def calculate_rmse(actuals, predictions):
    mse = mean_squared_error(actuals, predictions)
    return math.sqrt(mse)

# Load the best model for backtesting
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Predict the test set
with torch.no_grad():
    y_test_pred = model(X_test.cuda()).view(-1).cpu().numpy()
    y_test_actual = y_test.numpy()

# Calculate and print RMSE for backtesting
rmse = calculate_rmse(y_test_actual, y_test_pred)
print(f"Backtesting RMSE: {rmse}")

# Predicting the next "Close" value
X_most_recent_cuda = X_most_recent.cuda()
# Assuming X_most_recent is your most recent data point after appropriate preprocessing
with torch.no_grad():
    next_close_prediction = model(X_most_recent.cuda()).item()
    print(f"Prediction for the next 'Close' price: {next_close_prediction}")

   

