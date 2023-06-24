# Heavily inspired from
# https://github.com/instadeepai/jumanji/blob/main/jumanji/training/networks/bin_pack/actor_critic.py

from jumanji.environments.packing.bin_pack import BinPack, Observation
from jumanji.environments.packing.bin_pack.types import EMS, Item
from jumanji.training.networks.parametric_distribution import (
    FactorisedActionSpaceParametricDistribution,
)

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from typing import Optional, Sequence, Tuple
from collections import namedtuple

import torch 
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# representation:  s_0 = h(o_1, ..., o_t)
# Actor Critic:      p_k, v_k = f(s_k)

#class Representation(nn.Module):
class BinPackTorso(nn.Module):
  def __init__(self,
      num_layers: int,
      num_heads: int,
      key_size: int,
      mlp_units: Sequence[int],
      name: Optional[str] = None
  ):
    super(BinPackTorso, self).__init__()
    self.name = name
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.key_size = key_size
    self.mlp_units = mlp_units
    self.model_size = num_heads * key_size
  def __call__(self, observation: Observation) -> Tuple[chex.Array, chex.Array]:
    # EMS encoder
    #ems_mask = self._make_self_attention_mask(observation.ems_mask)
    ems_embeddings = self.embed_ems(observation.ems)

    # Item encoder
    #items_mask = self._make_self_attention_mask( observation.items_mask & ~observation.items_placed)
    items_embeddings = self.embed_items(observation.items)

    ## Decoder
    #ems_cross_items_mask = jnp.expand_dims(observation.action_mask, axis=-3)
    #items_cross_ems_mask = jnp.expand_dims(
    #    jnp.moveaxis(observation.action_mask, -1, -2), axis=-3
    #)
    #pass
    return ems_embeddings, items_embeddings


  def embed_ems(self, ems: EMS) -> np.ndarray:
    # Stack the 6 EMS attributes into a single vector [x1, x2, y1, y2, z1, z2].
    #https://dm-haiku.readthedocs.io/en/latest/api.html
    ems_leaves = torch.tensor( np.stack(jax.tree_util.tree_leaves(ems), axis=-1),dtype=torch.float32 )
    embeddings = nn.Linear(ems_leaves.shape[-1], self.model_size)(ems_leaves)  # Projection of the EMSs.
    return embeddings.detach().numpy()

  def embed_items(self, items: Item) -> np.ndarray:
    # Stack the 3 items attributes into a single vector [x_len, y_len, z_len].
    items_leaves = torch.tensor( np.stack(jax.tree_util.tree_leaves(items), axis=-1), dtype=torch.float32 )
    #https://dm-haiku.readthedocs.io/en/latest/api.html
    embeddings = nn.Linear(items_leaves.shape[-1], self.model_size)(items_leaves) # Projection of the EMSs.
    return embeddings.detach().numpy()

  def _make_self_attention_mask(self, mask: chex.Array) -> chex.Array:
    # Use the same mask for the query and the key.
    mask = jnp.einsum("...i,...j->...ij", mask, mask)
    # Expand on the head dimension.
    mask = jnp.expand_dims(mask, axis=-3)
    return mask

if __name__ == '__main__':
  env = BinPack()
  key = jax.random.PRNGKey(9)
  obs, timestep = env.reset(key)
  torso = BinPackTorso(num_layers=2,num_heads=8,key_size=16, mlp_units=[512], name="policy_torso")

  ems_embeddings, items_embeddings = torso(obs)
  pass
