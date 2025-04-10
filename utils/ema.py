#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import copy


class ModelEMA:
    """
    Model Exponential Moving Average
    
    Maintains moving averages of model parameters using an exponential decay.
    EMAs typically produce better validation and test metrics than normal
    training.
    
    Args:
        model (nn.Module): Model to apply EMA
        decay (float): Decay rate for EMA (higher means slower update)
        device (str): Device to store EMA model (if None, uses model's device)
    """
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        
        # Determine device
        if device is None or device == '':
            # Get device from model parameters
            self.device = next(model.parameters()).device
        else:
            self.device = device
            
        # Move EMA model to the correct device
        self.ema.to(self.device)
        
        # Parameters that will not be updated (e.g., batch norm statistics)
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        """
        Update the EMA parameters
        
        Args:
            model (nn.Module): Model to update from
        """
        with torch.no_grad():
            # Update all parameters
            for ema_param, param in zip(self.ema.parameters(), model.parameters()):
                if param.is_floating_point():
                    ema_param.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
            
            # Also update batch norm statistics
            self.update_attr(model)
    
    def update_attr(self, model):
        """
        Update attributes like 'running_mean' in batch norm layers
        
        Args:
            model (nn.Module): Model to update from
        """
        for ema_module, module in zip(self.ema.modules(), model.modules()):
            # Update batch norm running stats
            if hasattr(ema_module, 'running_mean') and hasattr(module, 'running_mean'):
                ema_module.running_mean.mul_(self.decay).add_(
                    module.running_mean, alpha=1 - self.decay
                )
                ema_module.running_var.mul_(self.decay).add_(
                    module.running_var, alpha=1 - self.decay
                )
                
            # Also update num_batches_tracked if present
            if hasattr(ema_module, 'num_batches_tracked') and hasattr(module, 'num_batches_tracked'):
                ema_module.num_batches_tracked.copy_(module.num_batches_tracked)