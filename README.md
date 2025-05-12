# SMP_using_IRRL
# Imports and params
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-3
REWARD_SCALE = 0.07

# DATASET - 1 APPLE STOCKS
class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.X, self.y = [], []
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len, 3])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



plt.grid(True)
plt.tight_layout()
plt.show()

print(f"MSE: {mean_squared_error(true_inv, preds_inv):.2f}")
print(f"MAE: {mean_absolute_error(true_inv, preds_inv):.2f}")
