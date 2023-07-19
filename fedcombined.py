import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define a simple model for illustration purposes
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Dummy datasets for two wireless devices
device1_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
device2_data = torch.tensor([[4.0], [5.0]], dtype=torch.float32)

# Create model instances for each device
device1_model_fedavg = SimpleModel()
device2_model_fedavg = SimpleModel()
device1_model_fedprox = SimpleModel()
device2_model_fedprox = SimpleModel()
device1_model_fedsgd = SimpleModel()
device2_model_fedsgd = SimpleModel()
device1_model_fedadaptive = SimpleModel()
device2_model_fedadaptive = SimpleModel()

# Define a loss function and optimizers for each algorithm
criterion = nn.MSELoss()
optimizer1_fedavg = optim.SGD(device1_model_fedavg.parameters(), lr=0.01)
optimizer2_fedavg = optim.SGD(device2_model_fedavg.parameters(), lr=0.01)
optimizer1_fedprox = optim.SGD(device1_model_fedprox.parameters(), lr=0.01)
optimizer2_fedprox = optim.SGD(device2_model_fedprox.parameters(), lr=0.01)
optimizer1_fedsgd = optim.SGD(device1_model_fedsgd.parameters(), lr=0.01)
optimizer2_fedsgd = optim.SGD(device2_model_fedsgd.parameters(), lr=0.01)
optimizer1_fedadaptive = optim.SGD(device1_model_fedadaptive.parameters(), lr=0.01)
optimizer2_fedadaptive = optim.SGD(device2_model_fedadaptive.parameters(), lr=0.01)

# Local training function for each device (FedAvg version)
def local_train_fedavg(model, optimizer, data):
    for _ in range(100):  # 100 local training iterations for simplicity
        outputs = model(data)
        loss = criterion(outputs, data)  # Use data as target for simplicity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Local training function for each device (FedProx version)
def local_train_fedprox(model, optimizer, data, prox_term):
    for _ in range(100):  # 100 local training iterations for simplicity
        outputs = model(data)
        loss = criterion(outputs, data)  # Use data as target for simplicity
        prox_loss = 0.5 * prox_term * sum((param**2).sum() for param in model.parameters())
        loss_with_prox = loss + prox_loss
        optimizer.zero_grad()
        loss_with_prox.backward()
        optimizer.step()

# Local training function for each device (FedSGD version)
def local_train_fedsgd(model, optimizer, data):
    for _ in range(100):  # 100 local training iterations for simplicity
        outputs = model(data)
        loss = criterion(outputs, data)  # Use data as target for simplicity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Local training function for each device (FedAdaptive version)
def local_train_fedadaptive(model, optimizer, data, lr):
    for _ in range(100):  # 100 local training iterations for simplicity
        outputs = model(data)
        loss = criterion(outputs, data)  # Use data as target for simplicity
        optimizer.param_groups[0]['lr'] = lr  # Update learning rate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Perform local training on each device with each algorithm
local_train_fedavg(device1_model_fedavg, optimizer1_fedavg, device1_data)
local_train_fedavg(device2_model_fedavg, optimizer2_fedavg, device2_data)
local_train_fedprox(device1_model_fedprox, optimizer1_fedprox, device1_data, prox_term=0.1)
local_train_fedprox(device2_model_fedprox, optimizer2_fedprox, device2_data, prox_term=0.1)
local_train_fedsgd(device1_model_fedsgd, optimizer1_fedsgd, device1_data)
local_train_fedsgd(device2_model_fedsgd, optimizer2_fedsgd, device2_data)
local_train_fedadaptive(device1_model_fedadaptive, optimizer1_fedadaptive, device1_data, lr=0.01)
local_train_fedadaptive(device2_model_fedadaptive, optimizer2_fedadaptive, device2_data, lr=0.001)

# Plot the datasets and the models after each algorithm
plt.figure(figsize=(10, 6))

# Plot device1's dataset
plt.scatter(device1_data, device1_data, label='Device 1 Data', color='blue')

# Plot device2's dataset
plt.scatter(device2_data, device2_data, label='Device 2 Data', color='red')

# Plot models after FedAvg
plt.plot(device1_data, device1_model_fedavg(device1_data).detach().numpy(), label='FedAvg (Device 1)', color='green', linestyle='--')
plt.plot(device2_data, device2_model_fedavg(device2_data).detach().numpy(), label='FedAvg (Device 2)', color='green')

# Plot models after FedProx
plt.plot(device1_data, device1_model_fedprox(device1_data).detach().numpy(), label='FedProx (Device 1)', color='purple', linestyle='--')
plt.plot(device2_data, device2_model_fedprox(device2_data).detach().numpy(), label='FedProx (Device 2)', color='purple')

# Plot models after FedSGD
plt.plot(device1_data, device1_model_fedsgd(device1_data).detach().numpy(), label='FedSGD (Device 1)', color='orange', linestyle='--')
plt.plot(device2_data, device2_model_fedsgd(device2_data).detach().numpy(), label='FedSGD (Device 2)', color='orange')

# Plot models after FedAdaptive
plt.plot(device1_data, device1_model_fedadaptive(device1_data).detach().numpy(), label='FedAdaptive (Device 1)', color='brown', linestyle='--')
plt.plot(device2_data, device2_model_fedadaptive(device2_data).detach().numpy(), label='FedAdaptive (Device 2)', color='brown')

plt.xlabel('Input Data')
plt.ylabel('Output Prediction')
plt.title('Federated Learning Algorithms in Wireless Communication')
plt.grid(True)
plt.legend()
plt.show()

