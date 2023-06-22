import jax
import jax.numpy as jnp

import jumanji
import chex

from jumanji.environments import BinPack
from jumanji.testing.env_not_smoke import SelectActionFn, check_env_does_not_smoke
from jumanji.environments.packing.bin_pack.types import ( Observation, State, item_from_space, location_from_space, )

import warnings
warnings.filterwarnings("ignore")

import numpy as np
env = BinPack()
key = jax.random.PRNGKey(9)
state, timestep = env.reset(key)


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
  key = jax.random.PRNGKey( jnp.random.randint(0,9999999) )
  #state, timestep = env.reset(key)
  state, timestep = jax.jit(env.reset)(key)
  while not done:
    env.render(state)
    action = random_policy(key, state)

    next_state, timestep = jax.jit(env.step)(state, action)

    is_action_valid = state.action_mask[tuple(action)]  
    done = ~jnp.any(next_state.action_mask) | ~is_action_valid 
    reward = env.reward_fn(state, action, next_state, is_action_valid , done)
    state = next_state

    score+=reward
    if done:
      scores.append(score)
      print(f"Episode {epi}, Return: {scores[-1]}")
      break
env.close()

