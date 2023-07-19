import torch
import torch.nn as nn
import torch.optim as optim

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
device1_model = SimpleModel()
device2_model = SimpleModel()

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer1 = optim.SGD(device1_model.parameters(), lr=0.01)
optimizer2 = optim.SGD(device2_model.parameters(), lr=0.01)

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

# Perform local training on each device with FedProx
local_train_fedprox(device1_model, optimizer1, device1_data, prox_term=0.1)
local_train_fedprox(device2_model, optimizer2, device2_data, prox_term=0.1)

# Federated Averaging with FedProx
def federated_averaging_fedprox(global_model, models, num_devices, prox_term):
    # Initialize the global model parameters with the average of the local models
    for param in global_model.parameters():
        param.data = sum(model_param.data for model_param in models.parameters()) / num_devices
    # Apply proximal term to the global model
    for param in global_model.parameters():
        param.data = param.data - prox_term * optimizer1.param_groups[0]['lr'] * (param.data - device1_model.state_dict()[list(device1_model.state_dict().keys())[0]])

# Initialize the global model with device1's model
global_model = SimpleModel()
global_model.load_state_dict(device1_model.state_dict())

# Perform Federated Averaging with FedProx using device2's model
federated_averaging_fedprox(global_model, device2_model, num_devices=2, prox_term=0.1)

# Print the updated global model's parameters
print("Updated Global Model Parameters:")
for name, param in global_model.named_parameters():
    print(name, param.data)

import matplotlib.pyplot as plt

# Plot device1's dataset
plt.scatter(device1_data, device1_data, label='Device 1 Data', color='blue')

# Plot device2's dataset
plt.scatter(device2_data, device2_data, label='Device 2 Data', color='red')

# Plot global model's parameters after FedProx
plt.plot(device1_data, global_model(device1_data).detach().numpy(), label='Global Model with FedProx', color='green')

plt.xlabel('Input Data')
plt.ylabel('Output Prediction')
plt.legend()
plt.title('Federated Proximal Gradient Descent (FedProx) in Wireless Communication')
plt.grid(True)
plt.show()
