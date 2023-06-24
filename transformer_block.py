# Heavily inspired from
# https://github.com/instadeepai/jumanji/blob/main/jumanji/training/networks/transformer_block.py
import numpy as np
import torch 
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Sequence

class TransformerBlock(nn.Module):
	"""Transformer block with post layer norm, implementing Attention Is All You Need [Vaswani et al., 2016]."""
	def __init__(self,
		num_heads: int,
		key_size: int,
		mlp_units: Sequence[int],
		w_init_scale: float,
		model_size: Optional[int] = None,
		name: Optional[str] = None,
    ):
		"""Initialises the transformer block module."""

		super(TransformerBlock, self).__init__()
		self.num_heads = num_heads
		self.key_size = key_size
		self.mlp_units = mlp_units
		#self.w_init = hk.initializers.VarianceScaling(w_init_scale)
		self.model_size = model_size or key_size * num_heads

	def __call__(self,
		query: np.ndarray,
		key: np.ndarray,
		value: np.ndarray,
		mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:


		"""
		# Multi-head attention and residual connection
		mha = hk.MultiHeadAttention(
				num_heads=self.num_heads,
				key_size=self.key_size,
				#w_init=self.w_init,
				model_size=self.model_size,
		)
		h = mha(query, key, value, mask) + query
		"""


		# Multi-head attention and residual connection
    multihead_attn = nn.MultiheadAttention(
			num_heads = self.num_heads
			kdim=None = self.key_size, 
			embed_dim = self.model_size, # Total dimension of the model.
		)
    h, _ = multihead_attn(query, key, value) +query
		h = nn.LayerNorm(embedding_dim)(h)


		"""
		h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

		# MLP and residual connection
		mlp = hk.nets.MLP((*self.mlp_units, self.model_size), activate_final=True)
		out = mlp(h) + h
		out = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(out)
		"""

		return out

