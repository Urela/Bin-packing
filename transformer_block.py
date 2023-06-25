# Heavily inspired from
# https://github.com/instadeepai/jumanji/blob/main/jumanji/training/networks/transformer_block.py
import numpy as np
import torch 
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
from typing import Optional, Sequence

class TransformerBlock(nn.Module):
  """Transformer block with post layer norm, implementing Attention Is All You Need [Vaswani et al., 2016]."""
  def __init__(self,
    num_heads: int,
    key_size: int,
    mlp_units: Sequence[int],
    model_size: Optional[int] = None,
    name: Optional[str] = None,
    ):
    """Initialises the transformer block module."""

    super(TransformerBlock, self).__init__()
    self.num_heads = num_heads
    self.key_size = key_size
    self.mlp_units = mlp_units
    self.model_size = model_size or key_size * num_heads
    self.name = name

  def __call__(self,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:

    print(self.key_size, self.model_size,key.shape, query.shape, value.shape)

    # Multi-head attention and residual connection
    multihead_attn = nn.MultiheadAttention(
      num_heads = self.num_heads,
      #kdim = self.key_size, 
      embed_dim = self.model_size # Total dimension of the model.
    )
    query, key, value = torch.tensor(query), torch.tensor(key), torch.tensor(value)
    h = multihead_attn(query, key, value)[0] + query
    h = nn.LayerNorm(h.shape)(h)

    #out = hk.nets.MLP((*self.mlp_units, self.model_size), activate_final=True)(h)+h
    #assert len(self.mlp_units)==1, "known issue, we need hk.nets.mlp but for torch, torchvision.ops.mlp?"
    #out = (lambda a,b: nn.Linear(*self.mlp_units, self.model_size)(a)+b)(h,h)
    #out = torchvision.ops.MLP(self.mlp_units[0], (*self.mlp_units[:1], self.model_size))(h)+h

    out = h
    out = nn.LayerNorm(out.shape)(out)
    return out

