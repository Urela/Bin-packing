# https://huggingface.co/InstaDeepAI/jumanji-benchmark-a2c-BinPack-v2
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

from model import random_policy, ActorCritic
agent = ActorCritic()

# rollout a few episodes
NUM_EPISODES = 10

states,scores = [], []
key = jax.random.PRNGKey(cfg.seed)
for episode in range(NUM_EPISODES):
    key = jax.random.PRNGKey( np.random.randint(0,9999999) )
    (key, reset_key), score = jax.random.split(key), 0
    state, timestep = jax.jit(env.reset)(reset_key)
    while not timestep.last():
      #env.render(state)
      key, action_key = jax.random.split(key)
      observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)

      action, _ = random_policy(env, observation, action_key)
      action = agent.get_action(observation)

      print("_state",_state.shape)

      next_state, timestep = jax.jit(env.step)(state, action)
      reward = timestep.reward
      states.append(next_state)
      score+=reward
      state = next_state
    # Freeze the terminal frame to pause the GIF.
    for _ in range(10): states.append(next_state)
    scores.append(score)
    print(f"Episode {episode}, Return: {scores[-1]}")

# animate a GIF
env.animate(states, interval=150).save("./binpack.gif")

# save PNG
import matplotlib.pyplot as plt
#%matplotlib inline
env.render(states[10])
plt.savefig("connector.png", dpi=300)
