import torch
import torch.nn as nn
from complextensor import ComplexTensor
from complex_ops import ComplexConvTranspose1d, complex_leaky_relu, complex_to_real

class ComplexDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims=[256, 128, 64], output_channels=1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Reverse hidden dimensions for decoder
        hidden_dims = list(reversed(hidden_dims))
        
        # Hidden layers with skip connections
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    ComplexConvTranspose1d(hidden_dims[i] * 2, hidden_dims[i+1], 
                                          kernel_size=5, stride=2, padding=2),
                    nn.LayerNorm([hidden_dims[i+1]]),
                )
            )
        
        # Output layer
        self.output_layer = ComplexConvTranspose1d(
            hidden_dims[-1] * 2, output_channels, kernel_size=7, stride=2, padding=3
        )
    
    def forward(self, x, encoder_features):
        """
        x: ComplexTensor from encoder of shape [batch_size, channels, length]
        encoder_features: list of features from encoder for skip connections
        
        Returns:
            output: Reconstructed audio signal [batch_size, output_channels, length]
        """
        # Reverse encoder features for skip connections
        encoder_features = list(reversed(encoder_features[:-1]))  # exclude last one which is same as x
        
        # Apply layers with skip connections
        for i, layer in enumerate(self.layers):
            # Concatenate with encoder features along channel dimension
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = layer(x)
            x = complex_leaky_relu(x)
        
        # Final output layer
        x = torch.cat([x, encoder_features[-1]], dim=1)
        x = self.output_layer(x)
        
        # Convert complex to real for final output
        return complex_to_real(x)