import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


data = pd.read_csv('data.csv')


X = data.drop(columns=['OS.time']).values
y = data['OS.time'].values


print(np.isnan(X).sum(), np.isnan(y).sum())
print(np.isinf(X).sum(), np.isinf(y).sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model = SimpleNN(X_train.shape[1])
model.apply(weights_init)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


best_test_loss = float('inf')
best_model_state = None


num_epochs = 10000
train_losses = []
test_losses = []
all_predictions = []
gradients = []
r2_scores = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    epoch_gradients = []
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for param in model.parameters():
            epoch_gradients.append(param.grad.abs().mean().item())
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    gradients.append(epoch_gradients)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
    
    model.eval()
    test_loss = 0.0
    predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predictions.append(outputs.numpy())
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    all_predictions.append(predictions)
    
    
    predictions_flat = np.concatenate(predictions).flatten()
    r2 = r2_score(y_test, predictions_flat)
    r2_scores.append(r2)
    print(f'Epoch {epoch+1}, R^2: {r2}')
    
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model.pth')
        print(f'Saved new best model at epoch {epoch+1} with test loss {test_loss}')





plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')


window_size = 50
train_losses_ma = pd.Series(train_losses).rolling(window=window_size).mean()
test_losses_ma = pd.Series(test_losses).rolling(window=window_size).mean()

plt.plot(range(1, num_epochs + 1), train_losses_ma, label='Train Loss (MA)', linestyle='--')
plt.plot(range(1, num_epochs + 1), test_losses_ma, label='Test Loss (MA)', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss with Moving Average')
plt.legend()
plt.savefig('train_test_loss.png')
plt.close()


final_predictions = np.array(all_predictions[-1]).flatten()
actuals = y_test_tensor.numpy().flatten()


correlation, p_value = pearsonr(actuals, final_predictions)
print(f'Pearson Correlation: {correlation}')
print(f'P-value: {p_value}')


plt.figure(figsize=(10, 5))
plt.scatter(actuals, final_predictions, color='blue', label=f'Predictions vs Actuals (r={correlation:.2f}, p={p_value:.2g})')
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual OS.time')
plt.ylabel('Predicted OS.time')
plt.title('Predictions vs Actuals')
plt.legend()
plt.savefig('predictions_vs_actuals.png')
plt.close()


errors = final_predictions - actuals
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30, color='purple', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.savefig('error_distribution.png')
plt.close()


actuals = y_test_tensor.numpy()
colors = cm.viridis(np.linspace(0, 1, num_epochs))

plt.figure(figsize=(10, 5))
plt.plot(actuals, label='Actual Values', color='b', marker='o', linestyle='-')

for i in range(0, num_epochs, max(1, num_epochs // 100)):
    predictions = np.array(all_predictions[i]).flatten()
    plt.plot(predictions, label=f'Epoch {i+1}', color=colors[i], linestyle='--')

plt.xlabel('Sample Index')
plt.ylabel('OS.time')
plt.title('Actual vs Predicted Values Over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('actual_vs_predicted_over_time.png')
plt.close()


for i, layer in enumerate(model.children()):
    if isinstance(layer, nn.Linear):
        plt.figure(figsize=(10, 5))
        plt.hist(layer.weight.detach().numpy().flatten(), bins=30, alpha=0.6, color='blue')
        plt.xlabel(f'Layer {i+1} Weights')
        plt.ylabel('Frequency')
        plt.title(f'Weight Distribution of Layer {i+1}')
        plt.savefig(f'layer_{i+1}_weight_distribution.png')
        plt.close()


importances = np.abs(model.fc1.weight.detach().numpy()).sum(axis=0)
indices = np.argsort(importances)

plt.figure(figsize=(10, 5))
plt.barh(range(X_train.shape[1]), importances[indices], align='center')
plt.xlabel('Importance')
plt.ylabel('Feature Index')
plt.title('Feature Importances in the First Layer')
plt.savefig('feature_importances.png')
plt.close()


for i, layer in enumerate(model.children()):
    if isinstance(layer, nn.Linear):
        plt.figure(figsize=(10, 5))
        plt.imshow(layer.weight.detach().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Weight Heatmap of Layer {i+1}')
        plt.xlabel('Input Features')
        plt.ylabel('Neurons')
        plt.savefig(f'layer_{i+1}_weight_heatmap.png')
        plt.close()


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), r2_scores, label='R^2 Score')


r2_scores_ma = pd.Series(r2_scores).rolling(window=window_size).mean()
plt.plot(range(1, num_epochs + 1), r2_scores_ma, label='R^2 Score (MA)', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('R^2 Score')
plt.title('R^2 Score over Epochs')
plt.legend()
plt.savefig('r2_over_epochs.png')
plt.close()
