from __future__ import annotations # delete 
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import jax
import chex
import jax.numpy as jnp

def flatten(obs: Observatoins):
  p=[]  # obs = timestep.observation
  p = np.append(obs.ems.x1, obs.ems.x2)
 #p = np.append(p,obs.ems.x2)
  p = np.append(p,obs.ems.y1)
  p = np.append(p,obs.ems.y2)
  p = np.append(p,obs.ems.z1)
  p = np.append(p,obs.ems.z2)
  p = np.append(p,obs.ems_mask.flatten())
  p = np.append(p,obs.items.x_len)
  p = np.append(p,obs.items.y_len)
  p = np.append(p,obs.items.z_len)
  p = np.append(p,obs.items_mask.flatten())
  p = np.append(p,obs.items_placed.flatten())
  return p


class Model(nn.Module):
  def __init__(self, state_size, action_size, hidden_size=128):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(state_size,  hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, action_size)

  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = F.relu(self.fc3(x)) 
    x = self.fc3(x)
    return x

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
