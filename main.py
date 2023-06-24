#from __future__ import annotations # delete 
import jax
import jax.numpy as jnp

import jumanji
import chex

from jumanji.environments import BinPack
from jumanji.environments.packing.bin_pack.types import EMS, Item
from jumanji.testing.env_not_smoke import SelectActionFn, check_env_does_not_smoke
from jumanji.environments.packing.bin_pack.types import ( Observation, State, item_from_space, location_from_space, )

import warnings
warnings.filterwarnings("ignore")
import numpy as np

env = BinPack()
key = jax.random.PRNGKey(9)
obs, timestep = env.reset(key)

import torch 
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
  def __init__(self, lr=0.001 ):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(781, 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, 2)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  """
  def forward(self, obs):

    ems_embeddings   = self.embed_ems(obs.ems)
    items_embeddings = self.embed_items(observation.items)
    x = obs

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    x = self.fc3(x)
    return x
    """

  def embed_ems(self, ems: EMS) -> chex.Array:
    # Stack the 6 EMS attributes into a single vector [x1, x2, y1, y2, z1, z2].
    ems_leaves = np.stack(jax.tree_util.tree_leaves(ems), axis=-1)
    embeddings = nn.Linear(self.model_size)(ems_leaves)  # Projection of the EMSs.
    return embeddings

  def embed_items(self, items: Item) -> chex.Array:
    # Stack the 3 items attributes into a single vector [x_len, y_len, z_len].
    items_leaves = np.stack(jax.tree_util.tree_leaves(items), axis=-1)
    embeddings = nn.Linear(self.model_size)(items_leaves) # Projection of the EMSs.
    return embeddings

agent = Model()

def random_policy(key: chex.PRNGKey, observation: Observation) -> chex.Array:
  """Randomly sample valid actions, as determined by `observation.action_mask`."""
  num_ems, num_items = np.asarray(env.action_spec().num_values)
  ems_item_id = jax.random.choice(
    key=key,
    a=num_ems * num_items,
    p=observation.action_mask.flatten(),
  )
  ems_id, item_id = jnp.divmod(ems_item_id, num_items)
  action = jnp.array([ems_id, item_id], jnp.int32)
  return action


scores = []
for epi in range(10):
  done, score = False,0
  key = jax.random.PRNGKey( np.random.randint(0,9999999) )
  #obs, timestep = jax.jit(env.reset)(key)
  obs, timestep = env.reset(key)
  while not done:
    #env.render(obs)
    action = random_policy(key, obs)

    #print("obs shape",obs)
    #ems_embeddings   = agent.embed_ems(obs.ems)
    #items_embeddings = agent.embed_items(obs.items)

    next_obs, timestep = jax.jit(env.step)(obs, action)

    is_action_valid = obs.action_mask[tuple(action)]  
    done = ~jnp.any(next_obs.action_mask) | ~is_action_valid 
    reward = env.reward_fn(obs, action, next_obs, is_action_valid , done)
    obs = next_obs

    score+=reward
    print(timestep.reward,) # what the hell is this reward??
    #print(timestep.observation)
    #print(f"Timestep: {timestep} | Reward: {reward}")
    if done:
      scores.append(score)
      print(f"Episode {epi}, Return: {scores[-1]}")
      break
env.close()

