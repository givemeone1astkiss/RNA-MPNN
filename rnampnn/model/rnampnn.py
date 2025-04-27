import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class RNAFeatures(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

class RNAMPNN(LightningModule):
    def __init__():
        super().__init__()
        
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def predict(self, *args, **kwargs):
        pass