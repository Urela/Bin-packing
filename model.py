
from __future__ import annotations # delete 

import gym
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


import jax
import jax.numpy as jnp
# *** Random policy ***
def random_policy(env: BinPack, observation: Observation, key: chex.PRNGKey) -> chex.Array:
  """Randomly sample valid actions, as determined by `observation.action_mask`."""
  num_ems, num_items = np.asarray(env.action_spec().num_values)
  ems_item_id = jax.random.choice(
    key=key,
    a=num_ems * num_items,
    p=observation.action_mask.flatten(),
  )
  ems_id, item_id = jnp.divmod(ems_item_id, num_items)
  action = jnp.array([ems_id, item_id], jnp.int32)
  return action, None


# *** PPO policy ***
class ActorCritic(nn.Module):
  def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
  def __init__(self, lr=0.001 ):
    super(ActorCritic, self).__init__()
    self.fc_actor = nn.Sequential(
      self.layer_init(nn.Linear(320, 128)), nn.ReLU(),
      self.layer_init(nn.Linear(128, 128)), nn.ReLU(),
      self.layer_init(nn.Linear(128,    2))
    )
    self.fc_critic = nn.Sequential(
      self.layer_init(nn.Linear(320, 128)), nn.ReLU(),
      self.layer_init(nn.Linear(128, 128)), nn.ReLU(),
      self.layer_init(nn.Linear(128,    1))
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
  def actor(self, state):
    dist = self.fc_actor(state)
    dist = Categorical(dist)
    return dist

  def critic(self, state):
    value = self.fc_critic(state)
    return value
  def get_action(self, observation: Observation):
    ems =  np.stack(jax.tree_util.tree_leaves(observation.items), axis=-1).flatten()
    items =  np.stack(jax.tree_util.tree_leaves(observation.ems), axis=-1).flatten()
    items_placed =  np.stack(jax.tree_util.tree_leaves(observation.items_placed), axis=-1).flatten()
    _state = torch.from_numpy(np.concatenate((ems, items, items_placed)))
    dist = self.actor(_state)
    action = dist.sample()

    probs  = torch.squeeze(dist.log_prob(action)).detach().numpy().astype(np.int32) 
    action = torch.squeeze(action).item().detach().numpy().astype(np.int32) 
    value  = torch.squeeze(value).item().detach().numpy().astype(np.int32) 


    #action = jnp.array(action, jnp.int32)
    return action

