import torch
import torch.nn as nn
import numpy as np

class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN Network that outputs quantile values for each action"""
    
    def __init__(self, input_channels=4, action_dim=18, num_quantiles=51):
        super(QRDQNNetwork, self).__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        
        # Convolutional layers for processing Atari frames
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        conv_out_size = self._get_conv_output_size(input_channels, 84, 84)
        
        # Fully connected layers - output quantiles for each action
        self.dense_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * num_quantiles)
        )
        
        # Fixed quantile fractions (tau values)
        self.register_buffer('quantile_fractions', 
                           torch.linspace(0.0, 1.0, num_quantiles + 2)[1:-1])
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, channels, height, width):
        x = torch.zeros(1, channels, height, width)
        x = self.conv_layers(x)
        return int(torch.numel(x) / x.shape[0])
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass returning quantile values
        
        Args:
            x: Input state tensor [batch_size, channels, height, width]
            
        Returns:
            Quantile values tensor [batch_size, action_dim, num_quantiles]
        """
        batch_size = x.shape[0]
        
        # Extract features
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        
        # Reshape to [batch_size, action_dim, num_quantiles]
        quantiles = x.view(batch_size, self.action_dim, self.num_quantiles)
        
        return quantiles
    
    def get_q_values(self, x):
        """
        Get Q-values by taking the mean over quantiles
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values tensor [batch_size, action_dim]
        """
        quantiles = self.forward(x)
        return quantiles.mean(dim=2)