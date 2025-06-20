import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pathlib import Path

# Load the breast cancer dataset from sklearn
df = load_breast_cancer()

# Split the data
X = torch.from_numpy(df.data).type(torch.float)
y = torch.from_numpy(df.target).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Make device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Create neural network class
class BreastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=30, out_features=75)   # Input layer
        self.layer_2 = nn.Linear(in_features=75, out_features=75)    # Hidden layer
        self.layer_3 = nn.Linear(in_features=75, out_features=1)   # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):   # x -> layer_1 -> layer_2 -> layer_3 -> output
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

# Create model
model = BreastCancerModel().to(device)

# Setup loss fn and optimizer
loss_fn = nn.BCEWithLogitsLoss()    # Sigmoid activation function built-in
optimizer = torch.optim.Adam(params=model.parameters(), # Adam optimizer
                             lr=0.001)

# Calculate accuracy - out of N samples, what percentage does it get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

# Set seeds
torch.manual_seed(27)
torch.mps.manual_seed(27)
np.random.seed(27)

# Set the number of epochs
epochs = 1000

# Put the data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training and evaluation loop
for epoch in range(epochs):
    # Training
    model.train()

    # 1. Forward pass
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # Turn logits -> pred probs -> pred labels

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)   # BCEWithLogits requires raw logits as input
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)    # Our custom accuracy function

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Print out what is happening
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# Save the model

# 1. Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "breast_cancer_model_0.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),  # Save only the model's state dict 
           f=MODEL_SAVE_PATH)

# Convert data using principled component analysis and plot
def pca_and_plot(model: torch.nn.Module, 
                 X: torch.Tensor, 
                 y_true: torch.Tensor, 
                 y_preds: torch.Tensor, 
                 title_1: str, 
                 title_2: str):
    """Performs principled component analysis on the data sets and plots them in 2D"""
    model.to("cpu")
    X, y_true, y_preds = X.to("cpu"), y_true.to("cpu"), y_preds.to("cpu")

    # Tranform data with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    scatter_1 = ax[0].scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=y_true, cmap="coolwarm", edgecolors="k")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title(title_1)

    handles_1, _ = scatter_1.legend_elements()
    ax[0].legend(handles_1, ["Malignant (0)", "Benign (1)"], loc="lower left")

    scatter_2 = ax[1].scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=y_preds, cmap="coolwarm", edgecolors="k")
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")
    ax[1].set_title(title_2)

    handles_2, _ = scatter_2.legend_elements()
    ax[1].legend(handles_2, ["Malignant (0)", "Benign (1)"], loc="lower left")

    plt.show()

# Final trained predictions
model.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model(X_test))).squeeze()

# Plot the data
pca_and_plot(model, X_test, y_test, y_preds, "PCA of Breast Cancer Test Dataset", "PCA of Model Test Predictions")