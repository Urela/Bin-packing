# https://huggingface.co/InstaDeepAI/jumanji-benchmark-a2c-BinPack-v2
from __future__ import annotations # delete 
import pickle
#import joblib

import numpy as np
import jax
import jax.numpy as jnp
from hydra import compose, initialize
from huggingface_hub import hf_hub_download

import warnings
warnings.filterwarnings("ignore")

from jumanji.training.setup_train import setup_agent, setup_env
from jumanji.training.utils import first_from_device

# initialise the config
with initialize(version_base=None, config_path="configs"):
  cfg = compose(config_name="config.yaml", overrides=["env=bin_pack", "agent=a2c"])


env = setup_env(cfg).unwrapped
agent = setup_agent(cfg, env)
#policy = jax.jit(agent.make_policy(params.actor, stochastic = False))
def policy(observation: Observation, key: chex.PRNGKey) -> chex.Array:
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

# rollout a few episodes
NUM_EPISODES = 1

obs_list,scores = [], []
key = jax.random.PRNGKey(cfg.seed)
for episode in range(NUM_EPISODES):
    key, reset_key = jax.random.split(key) 
    obs, timestep = jax.jit(env.reset)(reset_key)
    score = 0
    while not timestep.last():
      env.render(obs)
      key, action_key = jax.random.split(key)
      observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
      action, _ = policy(observation, action_key)
      next_obs, timestep = jax.jit(env.step)(obs, action)
      #reward = float(timestep.reward)
      reward = timestep.reward
      obs_list.append(next_obs)
      score+=reward
      obs = next_obs
    # Freeze the terminal frame to pause the GIF.
    for _ in range(10): 
      obs_list.append(next_obs)
    scores.append(score)
    print(f"Episode {episode}, Return: {scores[-1]}")

# animate a GIF
env.animate(obs_list, interval=150).save("./binpack.gif")

# save PNG
import matplotlib.pyplot as plt
#%matplotlib inline
env.render(obs_list[10])
plt.savefig("connector.png", dpi=300)
