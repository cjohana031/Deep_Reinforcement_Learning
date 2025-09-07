import torch.nn as nn
import torch

class DDQNNetwork(nn.Module):
    def __init__(self, input_channels=4, action_dim=18):
        super(DDQNNetwork, self).__init__()
        
        # Deeper convolutional layers for better feature extraction
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
        # For Atari (210, 160) -> (84, 84) after preprocessing
        conv_out_size = self._get_conv_output_size(input_channels, 84, 84)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
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
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x