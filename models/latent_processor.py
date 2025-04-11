import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LatentProcessor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Latent dimension from the encoder
        self.latent_dim = config['model']['bottleneck_channels']
        
        # Define condition dimension - can be configured or default to latent_dim
        self.condition_dim = config.get('model', {}).get('condition_dim', self.latent_dim)
        
        # BiGRU for temporal modeling as specified in the architecture
        # BiGRU(hidden=256) → [B,256]
        self.use_temporal = config.get('model', {}).get('use_temporal', True)
        if self.use_temporal:
            self.bigru = nn.GRU(
                input_size=self.latent_dim,
                hidden_size=self.latent_dim // 2,  # Divided by 2 because bidirectional
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        
        # MLP for condition processing
        # Following the parameter breakdown (~500K parameters for latent processing)
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, self.latent_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
        )
        
        # Optional VQ layer for discrete representations
        self.use_vq = config.get('model', {}).get('use_vq', False)
        if self.use_vq:
            self.num_embeddings = config.get('model', {}).get('num_embeddings', 512)
            self.vq_dim = self.latent_dim
            self.vq = VectorQuantizer(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.vq_dim,
                commitment_cost=0.25
            )
    
    def forward(self, z, condition=None):
        """
        Process the latent vector with optional conditioning and temporal modeling
        
        Args:
            z: Latent vector from encoder [B, latent_dim] or [B, T, latent_dim]
            condition: Conditioning vector [B, condition_dim] or None
            
        Returns:
            Processed latent vector [B, latent_dim]
        """
        batch_size = z.size(0)
        
        # Apply temporal modeling if enabled and input has temporal dimension
        if self.use_temporal:
            # If z doesn't have temporal dimension, add one
            if z.dim() == 2:
                z = z.unsqueeze(1)  # [B, latent_dim] -> [B, 1, latent_dim]
                
            # Apply BiGRU: Captures inter-frame dependencies as per architecture
            # h_t = BiGRU(z_t, h_{t-1})
            z, _ = self.bigru(z)
            
            # Take the last temporal output or mean across time
            if z.size(1) == 1:
                z = z.squeeze(1)  # [B, 1, latent_dim] -> [B, latent_dim]
            else:
                z = z.mean(dim=1)  # Average across temporal dimension
        
        # If no condition provided, return z unchanged or apply VQ if enabled
        if condition is None:
            if self.use_vq:
                vq_output = self.vq(z)
                z_q = vq_output['quantize']
                # Add VQ loss to the model for training
                self.vq_loss = vq_output['loss']
                return z_q
            return z
        
        # Process condition through MLP
        condition_embedding = self.condition_mlp(condition)
        
        # Apply conditioning: z' = z + MLP(condition)
        # Direct latent manipulation as per architecture
        z_conditioned = z + condition_embedding
        
        # Apply VQ if enabled
        if self.use_vq:
            vq_output = self.vq(z_conditioned)
            z_q = vq_output['quantize'] 
            # Add VQ loss to the model for training
            self.vq_loss = vq_output['loss']
            return z_q
        
        return z_conditioned
    
    def configure_optimizers(self):
        """Adam optimizer with learning rate from config"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate']
        )
        return optimizer


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer based on VQ-VAE paper
    (https://arxiv.org/abs/1711.00937)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return {
            'quantize': quantized,
            'loss': loss,
            'encodings': encodings,
            'encoding_indices': encoding_indices
        }