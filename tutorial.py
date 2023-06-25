import jax
import jax.numpy as jnp

import jumanji
import chex

from jumanji.environments import BinPack
from jumanji.testing.env_not_smoke import SelectActionFn, check_env_does_not_smoke
from jumanji.environments.packing.bin_pack.types import ( Observation, State, item_from_space, location_from_space, )

import warnings
warnings.filterwarnings("ignore")

env = BinPack()
import numpy as np
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
  #action = jnp.array([np.random.randint(0, env.obs_num_ems), np.random.randint(0, env.generator.max_num_items)])
  #action = jnp.array([np.random.randint(0, 20), np.random.randint(0, 20)])
  #action = env.action_spec().generate_value()
  return action


key = jax.random.PRNGKey(9)
#state, timestep = jax.jit(env.reset)(key)
state, timestep = env.reset(key)

scores = []
for epi in range(10):
  done, score = False,0
  #state, timestep = jax.jit(env.reset)(key)
  key = jax.random.PRNGKey( np.random.randint(0,9999999) )
  state, timestep = env.reset(key)
  while not done:
    env.render(state)
    action = random_policy(key, state)

    next_state, timestep = jax.jit(env.step)(state, action)

    ##'state', 'action', 'next_state', 'is_valid', and 'is_done'
    is_action_valid = state.action_mask[tuple(action)]  
    done = ~jnp.any(next_state.action_mask) | ~is_action_valid 
    reward = env.reward_fn(state, action, next_state, is_action_valid , done)
    state = next_state
    score+=reward

    if done:
      scores.append(score)
      #avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      #print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      print(f"Episode {epi}, Return: {scores[-1]}")
      break

env.close()

