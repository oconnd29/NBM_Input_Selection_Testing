import sys
import os
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from Utils.utils import TimingCallback
import numpy as np
import pandas as pd
from Utils.metrics_errors import point_metrics_with_nans, picp, mpiw, pinaw
import torch
import torch.nn as nn
from torch.autograd import Variable

"""Probablistic models"""

class SimpleFFNN_Pr(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleFFNN_Pr, self).__init__()
        
        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print('Device Found during model init:', self.device)

        # Input to first hidden layer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size).to(self.device))  # Move to device
        self.layers.append(nn.ReLU().to(self.device))  # Move to device
        
        # Add more hidden layers based on num_layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size).to(self.device))  # Move to device
            self.layers.append(nn.ReLU().to(self.device))  # Move to device
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_size, 1).to(self.device)  # For predicting the mean, moved to device
        self.fc_logvar = nn.Linear(hidden_size, 1).to(self.device)  # For predicting log variance, moved to device
    
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Pass through each hidden layer
        for layer in self.layers:
            x = layer(x)
        
        # Predict mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        
        # Calculate sigma and 95% confidence interval
        sigma = torch.exp(0.5 * log_var)
        lower = mu - 1.96 * sigma  # 95% lower bound (approx)
        upper = mu + 1.96 * sigma  # 95% upper bound (approx)
        
        return lower, mu, upper, log_var
        
class TemporalCNN_Pr(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, num_layers, output_size=1):
        super(TemporalCNN_Pr, self).__init__()
        
        # Set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print('Device Found during model init:', self.device)

        self.num_layers = num_layers
        self.relu = nn.ReLU().to(self.device)
        
        # Dynamically create a sequence of convolutional layers on the correct device
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else num_channels  # First layer uses input_size as in_channels
            conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=num_channels, 
                                   kernel_size=kernel_size, padding=(kernel_size - 1) // 2).to(self.device)
            layers.append(conv_layer)
        
        # Combine all layers into a ModuleList for multiple CNN layers
        self.conv_layers = nn.ModuleList(layers)
        
        # Fully connected layers for the mean and log variance
        self.fc_mu = nn.Linear(num_channels, output_size).to(self.device)
        self.fc_log_var = nn.Linear(num_channels, output_size).to(self.device)
    
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Reshape input for Conv1d (batch_size, input_channels, sequence_length)
        x = x.transpose(1, 2)  # Convert (batch_size, seq_length, input_size) -> (batch_size, input_size, seq_length)
        
        # Pass through each convolutional layer
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.relu(x)
        
        # Global average pooling to collapse the temporal dimension
        conv_out = x.mean(dim=2)  # Shape: (batch_size, num_channels)
        
        # Pass through fully connected layers
        mu = self.fc_mu(conv_out)
        log_var = self.fc_log_var(conv_out)
        
        # Compute standard deviation
        sigma = torch.exp(0.5 * log_var)
        
        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        
        return lower, mu, upper, log_var


class SimpleLSTM_Pr(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(SimpleLSTM_Pr, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print('Device Found during model init:', self.device)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        
        # Fully connected layer for mean output
        self.fc_mu = nn.Linear(hidden_size, output_size).to(self.device)
        
        # Fully connected layer for log variance output
        self.fc_log_var = nn.Linear(hidden_size, output_size).to(self.device)

        # Initialize random hidden and cell states using a normal distribution
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size).to(self.device), requires_grad=False)
        self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size).to(self.device), requires_grad=False)
    
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Repeat the random hidden and cell states to match the batch size
        h0_random = self.h0.repeat(1, x.size(0), 1)  # Already on the correct device
        c0_random = self.c0.repeat(1, x.size(0), 1)  # Already on the correct device
        
        # Pass through the LSTM layer using the random initial hidden and cell states
        lstm_out, (hn, cn) = self.lstm(x, (h0_random, c0_random))

        # Use the hidden state from the last time step
        final_output = lstm_out[:, -1, :]
                
        # Pass through the fully connected layers for mean and log variance
        mu = self.fc_mu(final_output)
        log_var = self.fc_log_var(final_output)

        # Compute the standard deviation
        sigma = torch.exp(0.5 * log_var)

        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return lower, mu, upper, log_var


class SimpleLSTM_Pr_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(SimpleLSTM_Pr, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device Found during model init:',self.device)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for mean output
        self.fc_mu = nn.Linear(hidden_size, output_size)
        
        # Fully connected layer for log variance output
        self.fc_log_var = nn.Linear(hidden_size, output_size)

        # Initialize random hidden and cell states using a normal distribution
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size), requires_grad=False)
        self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size), requires_grad=False)
    
    def forward(self, x):
      
        # Get the device of the input (automatically handles CPU or GPU)
        # device = x.device
        x = x.to(self.device)

        # Repeat the random hidden and cell states to match the batch size and move to the correct device
        h0_random = self.h0.repeat(1, x.size(0), 1).to(self.device).detach()  # Random initial hidden state
        c0_random = self.c0.repeat(1, x.size(0), 1).to(self.device).detach()  # Random initial cell state
        
        # Move LSTM parameters to the correct device
        self.lstm = self.lstm.to(self.device)
        
        # Pass through the LSTM layer using the random initial hidden and cell states
        lstm_out, (hn, cn) = self.lstm(x, (h0_random, c0_random))

        # Use the hidden state from the last time step
        final_output = lstm_out[:, -1, :]
                
        # Pass through the fully connected layers for mean and log variance
        mu = self.fc_mu(final_output).to(self.device)
        log_var = self.fc_log_var(final_output).to(self.device)

        # Compute the standard deviation
        sigma = torch.exp(0.5 * log_var)

        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return lower, mu, upper, log_var
        


        
class Groupwise_CNN_LSTM_Pr(nn.Module):
    def __init__(self, num_features, window_size, kernel_count, kernel_size, fc1_out_size, hidden_size_lstm, num_layers_lstm):
        super(Groupwise_CNN_LSTM_Pr, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        
        self.input_channels_CNN = window_size
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.total_out_size_CNN = window_size * kernel_count
        self.fc1_out_size = fc1_out_size
        self.hidden_size_lstm = hidden_size_lstm
        self.num_layers_lstm = num_layers_lstm
        self.num_features = num_features
        self.padding = 1

        # Define CNN Layer
        self.conv1d = nn.Conv1d(in_channels=self.input_channels_CNN, out_channels=self.total_out_size_CNN, kernel_size=self.kernel_size, padding=1, stride=1, groups=self.input_channels_CNN) 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=self.padding)
        self.fc1 = nn.Linear(self.kernel_count * ((self.num_features + (self.padding * 2)) // 2), self.fc1_out_size)

        # Define a list of LSTM layers, one for each variable
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=self.hidden_size_lstm, num_layers=self.num_layers_lstm, batch_first=True)
            for _ in range(self.fc1_out_size)
        ])
        
        # Output layers for probabilistic output (mean and log variance)
        self.linear_mu = nn.Linear(self.hidden_size_lstm * self.fc1_out_size, 1)  # For the mean estimate
        self.linear_logvar = nn.Linear(self.hidden_size_lstm * self.fc1_out_size, 1)  # For the log variance estimate

        # Initialize fixed zero hidden states
        self.h0 = torch.zeros(self.num_layers_lstm, 1, self.hidden_size_lstm)
        self.c0 = torch.zeros(self.num_layers_lstm, 1, self.hidden_size_lstm)
        
    def forward(self, x):

        """CNN Layer"""
        x = self.conv1d(x).to(x.device)
        x = self.relu(x).to(x.device)               
        x = self.maxpool(x).to(x.device)  # Reduce dimension across spatial domain

        x = x.view(x.size(0), self.input_channels_CNN, self.kernel_count, x.size(2)).to(x.device)  # Separate the tensor
        x = x.view(x.size(0), x.size(1), -1).to(x.device)  # Flatten across the spatial domain for fc layer
        x = self.fc1(x).to(x.device)  # Add a FC layer to each step in the window

        """LSTM Layer"""
        # List to store the outputs from each LSTM
        lstm_outs = []
        
        # Fixed zero hidden states (detached to prevent gradient updates)
        h0_fixed = self.h0.repeat(1, x.size(0), 1).detach()
        c0_fixed = self.c0.repeat(1, x.size(0), 1).detach()
        
        for i in range(self.fc1_out_size):
            # Get the i-th variable data (batch_size, seq_length, 1)
            x_i = x[:, :, i].unsqueeze(2)
            
            # Forward propagate through the i-th LSTM
            out, _ = self.lstms[i](x_i, (h0_fixed, c0_fixed))
            
            # Take the output from the last time step
            lstm_outs.append(out[:, -1, :])
        
        # Concatenate the outputs from each LSTM (batch_size, hidden_size_lstm * num_features)
        combined_out = torch.cat(lstm_outs, dim=1)
        
        # Pass through the fully connected layers to get mean (mu) and log variance (log_var)
        mu = self.linear_mu(combined_out).to(x.device)  # Mean estimate
        log_var = self.linear_logvar(combined_out).to(x.device)  # Log variance estimate
        
        # Compute standard deviation
        sigma = torch.exp(0.5 * log_var)  # Standard deviation
        
        # Compute the 95% confidence interval
        lower = mu - 1.96 * sigma  # 95% lower bound (approx)
        upper = mu + 1.96 * sigma  # 95% upper bound (approx)
        
        return lower, mu, upper, log_var
        
        


"""Deterministic Models"""

class GroupWise_CNN_LSTM(nn.Module):
    def __init__(self, num_features, window_size, kernel_count, kernel_size, fc1_out_size, hidden_size_lstm, num_layers_lstm):
        super(GroupWise_CNN_LSTM, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        
        self.input_channels_CNN = window_size  # equal to number of channels and groups, i.e., window size, CNN acts across spatial only and are separated to maintain temporal info.
        self.kernel_count = kernel_count              # number of kernels per group
        self.kernel_size = kernel_size                # kernel size
        self.total_out_size_CNN = window_size * kernel_count    # tensor size for all CNNs (all groups)
        self.fc1_out_size = fc1_out_size              # output size for the first dense layer, will be equal to the number of separate LSTMs
        self.hidden_size_lstm = hidden_size_lstm
        self.num_layers_lstm = num_layers_lstm
        self.num_features = num_features
        self.padding = 1

        # Define CNN Layer
        self.conv1d = nn.Conv1d(in_channels=self.input_channels_CNN, out_channels=self.total_out_size_CNN, kernel_size=self.kernel_size, padding=1, stride=1, groups=self.input_channels_CNN) 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=self.padding)
        self.fc1 = nn.Linear(self.kernel_count * ((self.num_features + (self.padding * 2)) // 2), self.fc1_out_size)

        # Define a list of LSTM layers, one for each variable
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=self.hidden_size_lstm, num_layers=self.num_layers_lstm, batch_first=True)   # Input size is the number of values to take in for each pass through LSTM
            for _ in range(self.fc1_out_size)
        ])
        
        # Define the output layer
        self.fc2 = nn.Linear(self.hidden_size_lstm * self.fc1_out_size, 1)

        # Initialize fixed zero hidden states
        self.h0 = torch.zeros(self.num_layers_lstm, 1, self.hidden_size_lstm)
        self.c0 = torch.zeros(self.num_layers_lstm, 1, self.hidden_size_lstm)
        
    def forward(self, x):

        """CNN Layer"""
        x = self.conv1d(x).to(x.device)
        x = self.relu(x).to(x.device)               
        x = self.maxpool(x).to(x.device)  # Reduce dimension across spatial domain

        x = x.view(x.size(0), self.input_channels_CNN, self.kernel_count, x.size(2)).to(x.device)  # Separate the tensor
        x = x.view(x.size(0), x.size(1), -1).to(x.device)  # Flatten across the spatial domain for fc layer
        x = self.fc1(x).to(x.device)  # Add a FC layer to each step in the window

        """LSTM Layer"""
        # List to store the outputs from each LSTM
        lstm_outs = []
        
        # Fixed zero hidden states (detached to prevent gradient updates)
        h0_fixed = self.h0.repeat(1, x.size(0), 1).detach()
        c0_fixed = self.c0.repeat(1, x.size(0), 1).detach()
        
        for i in range(self.fc1_out_size):
            # Get the i-th variable data (batch_size, seq_length, 1)
            x_i = x[:, :, i].unsqueeze(2)
            
            # Forward propagate through the i-th LSTM
            out, _ = self.lstms[i](x_i, (h0_fixed, c0_fixed))
            
            # Take the output from the last time step
            lstm_outs.append(out[:, -1, :])
        
        # Concatenate the outputs from each LSTM (batch_size, hidden_size_lstm * num_features)
        combined_out = torch.cat(lstm_outs, dim=1)
        
        # Pass through the fully connected layer
        out = self.fc2(combined_out).to(x.device)
        
        return out

"""OLd models"""
class TemporalCNN_Pr_oneLayer(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, output_size=1):
        super(TemporalCNN_Pr_oneLayer, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # 1D Convolutional layer (for temporal data)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, 
                               kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Fully connected layer for mean output
        self.fc_mu = nn.Linear(num_channels, output_size)
        
        # Fully connected layer for log variance output
        self.fc_log_var = nn.Linear(num_channels, output_size)
    
    def forward(self, x):
        # Ensure all layers and operations are on the same device as the input
        device = x.device
        
        # Reshape input for Conv1d (batch_size, input_channels, sequence_length)
        x = x.transpose(1, 2)  # Convert (batch_size, seq_length, input_size) -> (batch_size, input_size, seq_length)
        
        # Pass through Conv1d layer
        conv_out = self.conv1(x)
        conv_out = self.relu(conv_out)
        
        # Global average pooling to collapse the temporal dimension (optional, but helps with overfitting)
        conv_out = conv_out.mean(dim=2)  # Shape: (batch_size, num_channels)
        
        # Pass through fully connected layers
        mu = self.fc_mu(conv_out)
        log_var = self.fc_log_var(conv_out)
        
        # Compute standard deviation
        sigma = torch.exp(0.5 * log_var)
        
        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        
        return lower, mu, upper, log_var 
        
class SimpleLSTM_Pr_fixed(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(SimpleLSTM_Pr_fixed, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for mean output
        self.fc_mu = nn.Linear(hidden_size, output_size)
        
        # Fully connected layer for log variance output
        self.fc_log_var = nn.Linear(hidden_size, output_size)

        # Initialize fixed zero hidden states
        self.h0 = torch.zeros(num_layers, 1, hidden_size)
        self.c0 = torch.zeros(num_layers, 1, hidden_size)
    
    def forward(self, x):
      
        # Get the device of the input (automatically handles CPU or GPU)
        device = x.device

        # Repeat the fixed hidden and cell states to match the batch size and move to the correct device
        h0_fixed = self.h0.repeat(1, x.size(0), 1).to(device).detach()  # Fixed initial hidden state
        c0_fixed = self.c0.repeat(1, x.size(0), 1).to(device).detach()  # Fixed initial cell state
        
        # Move LSTM parameters to the correct device
        self.lstm = self.lstm.to(device)
        
        # Pass through the LSTM layer using the fixed initial hidden and cell states
        lstm_out, (hn, cn) = self.lstm(x, (h0_fixed, c0_fixed))

        # Use the hidden state from the last time step
        final_output = lstm_out[:, -1, :]
                
        # Pass through the fully connected layers for mean and log variance
        mu = self.fc_mu(final_output).to(device)
        log_var = self.fc_log_var(final_output).to(device)

        # Compute the standard deviation
        sigma = torch.exp(0.5 * log_var)

        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return lower, mu, upper, log_var
class SimpleNN_Pr(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN_Pr, self).__init__()
        self.linear_mu = nn.Linear(input_dim, 1)  # For the mean estimate
        self.linear_logvar = nn.Linear(input_dim, 1)  # For the log variance
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)  # Mean estimate
        log_var = self.linear_logvar(x)  # Log variance estimate
        
        sigma = torch.exp(0.5 * log_var)  # Standard deviation
        
        lower = mu - 1.96 * sigma  # 95% lower bound (approx)
        upper = mu + 1.96 * sigma  # 95% upper bound (approx)
        
        return lower, mu, upper, log_var
        
class SpatialCNN_Pr(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, output_size=1):
        super(SpatialCNN_Pr, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # 1D Convolutional layer to operate across spatial features (input_size is spatial dimension)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, 
                               kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        # Activation function
        self.relu = nn.ReLU()

        # Fully connected layer for mean output
        self.fc_mu = nn.Linear(num_channels, output_size)
        
        # Fully connected layer for log variance output
        self.fc_log_var = nn.Linear(num_channels, output_size)

    def forward(self, x):
        # Ensure all layers and operations are on the same device as the input
        device = x.device

        # Reshape input to (batch_size, input_size, seq_length) to treat spatial dimension (10 features) as channels
        # Input: (batch_size, seq_length, input_size) -> (batch_size, input_size, seq_length)
        x = x.transpose(1, 2)

        # Apply convolution across the spatial dimension (input_size, i.e., 10 features)
        conv_out = self.conv1(x)
        conv_out = self.relu(conv_out)

        # Optional: Global average pooling across the sequence dimension (seq_length, i.e., 18 time steps)
        conv_out = conv_out.mean(dim=2)  # Shape: (batch_size, num_channels)

        # Pass through fully connected layers
        mu = self.fc_mu(conv_out)
        log_var = self.fc_log_var(conv_out)

        # Compute standard deviation
        sigma = torch.exp(0.5 * log_var)

        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return lower, mu, upper, log_var
        

class TemporalSpatial_CNN_Pr(nn.Module):
    def __init__(self, num_features, window_size, num_channels, kernel_size=(3, 3), output_size=1):
        super(TemporalSpatial_CNN_Pr, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # 2D Convolutional layer to operate across both time and feature dimensions
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, 
                               kernel_size=kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))

        # Activation function
        self.relu = nn.ReLU()

        # Fully connected layer for mean output
        self.fc_mu = nn.Linear(num_channels, output_size)
        
        # Fully connected layer for log variance output
        self.fc_log_var = nn.Linear(num_channels, output_size)

    def forward(self, x):
        # Ensure all layers and operations are on the same device as the input
        device = x.device

        # Add a channel dimension to the input (batch_size, 1, window_size, num_features)
        x = x.unsqueeze(1)  # Input shape: (batch_size, 1, window_size, num_features)

        # Apply 2D convolution across the spatial and temporal dimensions
        conv_out = self.conv1(x)
        conv_out = self.relu(conv_out)

        # Global average pooling across the time and feature dimensions
        conv_out = conv_out.mean(dim=[2, 3])  # Shape: (batch_size, num_channels)

        # Pass through fully connected layers
        mu = self.fc_mu(conv_out)
        log_var = self.fc_log_var(conv_out)

        # Compute standard deviation
        sigma = torch.exp(0.5 * log_var)

        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return lower, mu, upper, log_var

class SimpleLSTM_2(nn.Module):
    def __init__(self, input_features=20, window_size=10, hidden_dim=32, output_dim=1):
        super(SimpleLSTM_2, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        self.input_features = input_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        # Define a separate LSTM for each feature and move each one to the device
        self.lstms = nn.ModuleList([nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True).to(self.device) for _ in range(input_features)])
        
        # # Fully connected layer to combine the last hidden states of all features into one output
        # self.fc = nn.Linear(input_features * hidden_dim, output_dim)
        
        # Fully connected layer for mean output
        self.fc_mu = nn.Linear(input_features * hidden_dim, output_dim).to(self.device)
        
        # Fully connected layer for log variance output
        self.fc_log_var = nn.Linear(input_features * hidden_dim, output_dim).to(self.device)

    def forward(self, x):
        # x shape: (batch_size, window_size, input_features)
        batch_size = x.size(0)

        # Pass X to the device
        x = x.to(self.device)

        # Convert X so it is in form ((batch_size, input_features, window_size))
        x = x.transpose(1, 2)

        # Process each feature with its corresponding LSTM
        lstm_outputs = []

        for i in range(self.input_features):
            # Select the i-th feature across all batch samples and time steps: (batch_size, window_size, 1)
            feature_seq = x[:, i, :].unsqueeze(-1)

            # Pass the feature through its corresponding LSTM
            lstm_out, (h_n, c_n) = self.lstms[i](feature_seq)

            # Collect the last hidden state of the LSTM (h_n[-1] is the last hidden state for the last time step)
            lstm_outputs.append(h_n[-1])

        # Concatenate the last hidden states from all features: (batch_size, input_features * hidden_dim)
        combined_output = torch.cat(lstm_outputs, dim=1)

        # # Pass through the fully connected layer to get the final output: (batch_size, output_dim)
        # output = self.fc(combined_output)
        
        # Pass through the fully connected layers for mean and log variance
        mu = self.fc_mu(combined_output)
        log_var = self.fc_log_var(combined_output)

        # Compute the standard deviation
        sigma = torch.exp(0.5 * log_var)

        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return lower, mu, upper, log_var
        
class SimpleLSTM_3(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, output_size=1):
        super(SimpleLSTM_3, self).__init__()
        
        # Device setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.device = device
        
        # LSTM layer (takes input_size as the input features, hidden_dim as the number of LSTM units)
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, output_size)
        self.fc_log_var = nn.Linear(hidden_dim, output_size)


    def forward(self, x):
        # Ensure x is on the right device
        x = x.to(self.device)
        
        # Pass the input through LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out_last_step = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Pass through the fully connected layers
        mu = self.fc_mu(lstm_out_last_step)
        log_var = self.fc_log_var(lstm_out_last_step)
        
        # Compute standard deviation
        sigma = torch.exp(0.5 * log_var)
        
        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        
        return lower, mu, upper, log_var

class SimpleLSTM_4(nn.Module):
    def __init__(self, input_size=20, hidden_dim=51, output_size=1):
        super(SimpleLSTM_4, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Two LSTM layers: First processes input features, second processes the output of the first layer
        self.lstm1 = nn.LSTMCell(input_size, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, output_size)
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(output_size, output_size)
        self.fc_log_var = nn.Linear(output_size, output_size)

    def forward(self, input, future=0):
        # List to collect outputs for each time step
        hidden_states = []

        # Initialize hidden and cell states for LSTM1 and LSTM2
        h_t = torch.zeros(input.size(0), self.hidden_dim).to(input.device)
        c_t = torch.zeros(input.size(0), self.hidden_dim).to(input.device)
        h_t2 = torch.zeros(input.size(0), 1).to(input.device)
        c_t2 = torch.zeros(input.size(0), 1).to(input.device)

        # Loop over each time step and feature to process input
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):  # input.chunk splits input into timesteps
            input_t = input_t.squeeze(1)  # Remove the extra dimension for batch processing
            
            # LSTM1 processes the input feature-wise and outputs hidden states
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            
            # LSTM2 processes the hidden state from LSTM1
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            
            # Store the hidden states of LSTM2
            hidden_states.append(h_t2)

        # If future prediction is required, continue generating future outputs
        for i in range(future):
            h_t, c_t = self.lstm1(h_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            hidden_states.append(h_t2)

        # Stack the hidden states across time steps: (batch_size, seq_length, hidden_dim)
        hidden_states = torch.stack(hidden_states, dim=1)

        # Aggregate information across all time steps
        # You can choose to sum, mean, or concatenate the hidden states:
        aggregated_states = hidden_states.mean(dim=1)  # (batch_size, output_size)
        # Other options: 
        # aggregated_states = hidden_states.sum(dim=1)  # or sum the hidden states
        # aggregated_states = hidden_states.view(hidden_states.size(0), -1)  # or flatten and concatenate
        
        # Pass the aggregated states through the fully connected layers for probabilistic output
        mu = self.fc_mu(aggregated_states)       # Shape: (batch_size, output_size)
        log_var = self.fc_log_var(aggregated_states)  # Shape: (batch_size, output_size)
        
        # Compute standard deviation from log variance
        sigma = torch.exp(0.5 * log_var)

        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma

        return lower, mu, upper, log_var

class GroupWise_LSTM_Pr(nn.Module):
    def __init__(self, num_variables, hidden_size, num_layers):
        super(GroupWise_LSTM_Pr, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_variables = num_variables

        # Define a list of LSTM layers, one for each variable (feature)
        self.lstms = nn.ModuleList([
            # input size takes in 1 value per step, hidden_size is the number of lstm cells per feature, nu  layer is num lstm layers,
            nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
            for _ in range(self.num_variables)
        ])
        
        # Two fully connected layers to predict mean (mu) and log variance (log_var)
        self.fc_mu = nn.Linear(self.hidden_size * self.num_variables, 1)
        self.fc_logvar = nn.Linear(self.hidden_size * self.num_variables, 1)

    def forward(self, x):
        # Ensure input is on the right device
        x = x.to(next(self.parameters()).device)

        # List to store the outputs from each LSTM
        lstm_outs = []
        
        # Initialize hidden states for each LSTM
        h0_fixed = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).detach()
        c0_fixed = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).detach()
        
        # Iterate through each feature
        for i in range(self.num_variables):
            # Get the i-th feature data as a time series (batch_size, seq_length, 1)
            x_i = x[:, :, i].unsqueeze(2)
            
            # Forward propagate through the i-th LSTM
            out, _ = self.lstms[i](x_i, (h0_fixed, c0_fixed))
            
            # Take the output from the last time step
            lstm_outs.append(out[:, -1, :])  # Shape (batch_size, hidden_size)
        
        # Concatenate the outputs from each LSTM (batch_size, hidden_size * num_variables)
        combined_out = torch.cat(lstm_outs, dim=1)
        
        # Predict mean (mu) and log variance (log_var)
        mu = self.fc_mu(combined_out)
        log_var = self.fc_logvar(combined_out)
        
        # Compute standard deviation
        sigma = torch.exp(0.5 * log_var)
        
        # Calculate 95% confidence interval bounds
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        
        return lower, mu, upper, log_var