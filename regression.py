import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import matplotlib.pyplot as plt
from data.stock_data import pull_single_stock

# Load the dataset
df = pull_single_stock("TSLA")

# Dropping irrelevant columns, assumed based on domain knowledge
df = df.drop(['Date', 'Ticker'], axis=1)

# Handling inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Sophisticated imputation
imputer = IterativeImputer()
df_imputed = imputer.fit_transform(df)
df = pd.DataFrame(df_imputed, columns=df.columns)

# Feature-target split
X = df.drop('Close', axis=1).values
y = df['Close'].values

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# DataLoader for training data
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

class AdvancedRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(AdvancedRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Increased width
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)  # Slightly increased dropout

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

# Note: Adapt the input_size and ensure appropriate preprocessing
model = AdvancedRegressionModel(input_size=X_train.shape[1])
model.cuda()  # Move model to GPU

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training loop
epochs = 500
train_losses, val_losses = [], []
early_stopping_patience = 20
early_stopping_counter = 0
best_val_loss = np.inf

for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())

    # Validation phase
    model.eval()
    with torch.no_grad():
        inputs, targets = X_test.cuda(), y_test.cuda()
        outputs = model(inputs)
        val_loss = criterion(outputs, targets.unsqueeze(1))
        val_losses.append(val_loss.item())
    
    scheduler.step()

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# Plotting the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()  
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Extract the last row of data (most recent data point)
    last_row_scaled = X_scaled[-1].reshape(1, -1)  # Ensure it's a 2D array for a single sample
    last_row_tensor = torch.FloatTensor(last_row_scaled).cuda()  # Convert to tensor and move to GPU
    
    # Make prediction
    prediction = model(last_row_tensor)
    print(f"Predicted next day's 'Close' price: {prediction.item()}")


