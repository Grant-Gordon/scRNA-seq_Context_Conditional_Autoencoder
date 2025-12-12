#context-classifier.py
import torch
import torch.nn as nn
from torch import Tensor 
from typing import Dict, Tuple, List


class ContextClassifier(nn.Module):

    def __init__(self, gene_dim: int, latent_dim:int, field_specs: Dict[str, int]):
        super().__init__()
        self.fields = list(field_specs.keys())
        self.latent_dim = latent_dim
        self.shared = nn.Sequential(
            nn.Linear(gene_dim, self.latent_dim),
            nn.ReLU(inplace=True)
        )
        self.heads = nn.ModuleDict({
            f: nn.Linear(self.latent_dim, field_specs[f]["cardinality"]) for f in self.fields
        })

    def forward(self, x:torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {f: self.heads[f](h) for f in self.fields} # per field logits
    
