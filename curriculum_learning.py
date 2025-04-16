# Imort necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go

# Define a simple neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10) # Input layer (1) -> Hidden layer (10)
        self.fc2 = nn.Linear(10, 1) # Hidden layer(10) -> Output layer (1)
    
    def forward (self, x):
        x = torch.relu(self.fc1(x)) # Activation function for hidden layer
        x = self.fc2(x)
        return x

# Define a custom dataset class for curruculum learning
class CurriculumDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Generate sample data for training
np.random.seed(0)
data = np.random.uniform(-10, 10, size=(1000, 1))
labels = data ** 2 # Simple regression task

# Sort data by difficulty (in this case , by input value)
idx = np.argsort(np.abs(data).flatten())
data = data[idx]
labels = labels[idx]

# Create dataset and data loader
dataset = CurriculumDataset(data, labels)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define curriculum learning function
def currriculum_learning(model, data_loader, epochs, criterion, optimizer):
    # Sort data by difficulty (in this case, by input value)
    ##sorted_data_loader = sorted(data_loader, key=lambda x: x[0].item())

    epoch_losses = []
    epoch_accuracies = []

    # Train model in stages, increasing difficulty
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in data_loader:
            # Unpack batched data
            inputs, labels = batch

            # Convert inputs and labels to float tensors
            inputs = inputs.float()
            labels = labels.float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track epoch metrics
            epoch_loss += loss.item()
            epoch_accuracy += torch.sum(torch.isclose(outputs, labels.float(), atol=0.1)).item()

            # Track epoch losses and accuracies
            epoch_losses.append(epoch_loss / len(data_loader))
            epoch_accuracies.append(epoch_accuracy / len(data_loader) / batch_size)

            # Print loss and accuracy at each stage
            #if epoch == 0:
                #print(f"Stage {epoch + 1}")
                #print(f'Loss: {loss.item()}')
                #print(f'Accuracy: {torch.sum(torch.isclose(outputs, labels, atol=0.1)).item() / batch_size}')
                #print('---')
            
            # Print epoch metrics
            print(f'Epoch {epoch + 1} Metrics:')
            print(f'Average Loss: {epoch_loss / len(data_loader )}')
            print(f"Average Accuracy: {epoch_accuracy / len(data_loader) / batch_size}")
            print('===')
        
    return epoch_losses, epoch_accuracies

# Initialize model, criterion, and optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train model using curriculum learning
result = currriculum_learning(model, data_loader, epochs=10, criterion=criterion, optimizer=optimizer)

# Unpack Result
losses, accuracies = result
# Plot epoch losses and accuracies
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, len(losses) + 1)), y = losses, name='Loss'))
fig.add_trace(go.Scatter(x=list(range(1,len(accuracies) + 1)), y= accuracies, name='Accuracy'))
fig.update_layout(title='Curriculum Learning Metrics', xaxis_title='Epoch', yaxis_title='Value')
fig.show()

# Save output to a file
import pandas as pd

output_data = {'Epoch': range(1, len(losses) + 1), 'Loss': losses,'Accuracy': accuracies}
df = pd.DataFrame(output_data)
df.to_csv('output.csv', index=False)


