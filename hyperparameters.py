import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data = pd.read_csv('data.csv')


X = data.drop(columns=['OS.time']).values
y = data['OS.time'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        last_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(last_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        self.layers.append(nn.Linear(last_dim, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def objective(trial):
    
    num_layers = trial.suggest_int('num_layers', 2, 5)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    momentum = trial.suggest_float('momentum', 0.5, 0.9)
    num_epochs = trial.suggest_int('num_epochs', 6000, 10000)

    
    model = SimpleNN(X_train.shape[1], hidden_dim, num_layers, dropout_rate).to(device)
    model.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=momentum)

    
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item() * inputs.size(0)
        
        test_loss /= len(test_loader.dataset)

        trial.report(test_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return test_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)


print(f"Best trial parameters: {study.best_trial.params}")
print(f"Best trial test loss: {study.best_trial.value}")


import optuna.visualization as vis

vis.plot_param_importances(study).show()

vis.plot_parallel_coordinate(study).show()

best_params = study.best_trial.params

model = SimpleNN(X_train.shape[1], best_params['hidden_dim'], best_params['num_layers'], best_params['dropout_rate']).to(device)
model.apply(weights_init)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=best_params['momentum'])

test_losses = []
for epoch in range(best_params['num_epochs']):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item() * inputs.size(0)
    
    test_loss /= len(test_loader.dataset)

    if epoch % 100 == 0:
        test_losses.append(test_loss)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss}')

print("Training completed with best hyperparameters.")


plt.figure(figsize=(10, 5))
plt.plot(range(1, len(test_losses) * 100, 100), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss over Epochs')
plt.legend()
plt.show()
