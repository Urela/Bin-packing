from __future__ import annotations # delete 
import warnings  
warnings.filterwarnings('ignore') 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from jumanji.wrappers import AutoResetWrapper
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import jumanji
from networks import flatten
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = jumanji.make("BinPack-v2")
env = AutoResetWrapper(env)
dummy_obs = env.observation_spec().generate_value()
dummy_obs = flatten(env.observation_spec().generate_value())
num_ems, num_items = env.action_spec().num_values


class Model(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=128):
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

class DQN:
  def __init__(self, in_size, out_size):
    self.lr = 1e-3
    self.gamma   = 0.99
    self.epsilon = 1.0
    self.eps_min = 0.05
    self.eps_dec = 5e-4

    self.out_size = out_size  
    self.memory = deque(maxlen=25000)
    self.policy = Model(in_size, out_size).to(device)
    self.target = Model(in_size, out_size).to(device) 
    self.target.load_state_dict(self.policy.state_dict())
    self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

  def update_target(self):
    self.target.load_state_dict( self.policy.state_dict() )

  def update_epsilon(self):
    self.epsilon = max(self.eps_min, self.epsilon*self.eps_dec)

  def get_action(self, observation, timestep):
    if np.random.rand() >= self.epsilon:
      with torch.no_grad():
        observation = torch.FloatTensor(observation).to(device)
        ems_item_id = self.policy(observation ).max(0)[1].view(-1)
        ems_item_id = int(ems_item_id)
        ems_id, item_id = jnp.divmod(ems_item_id, num_items)
        action = jnp.array([ems_id, item_id])

    else: # random policy
      num_ems, num_items = np.asarray(env.action_spec().num_values)
      ems_item_id = jax.random.choice(
        key=jax.random.PRNGKey( np.random.randint(0,9999999) ),
        a=num_ems * num_items,
        p=timestep.observation.action_mask.flatten(),

      )
      ems_id, item_id = jnp.divmod(ems_item_id, num_items)
      action = jnp.array([ems_id, item_id], jnp.int32)
    return action

  def train(self, batch_size=3):
    if len(self.memory) >= batch_size:
      print("training")
      for i in range(10):
        batch = random.sample(self.memory, batch_size)
        states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
        actions = torch.tensor([[x[1]] for x in batch])
        rewards = torch.tensor([[x[2]] for x in batch]).float()
        nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
        dones   = torch.tensor([x[4] for x in batch])

        q_pred = self.policy(states).gather(1, actions)
        q_targ = self.target(nstates).max(1)[0].unsqueeze(1)
        q_targ[dones] = 0.0  # set all terminal states' value to zero
        q_targ = rewards + self.gamma * q_targ

        loss = F.smooth_l1_loss(q_pred, q_targ).to('cpu')
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()
    pass

action_size =  num_ems * num_items
state_size = dummy_obs.shape[0]
agent = DQN(state_size, action_size)

NUM_EPISODES = 10
states, scores = [], []
key = jax.random.PRNGKey(1337)
for episode in range(NUM_EPISODES):
    key = jax.random.PRNGKey( np.random.randint(0,9999999) )
    (key, reset_key), score = jax.random.split(key), 0
    state, timestep = jax.jit(env.reset)(reset_key)
    while not timestep.last():
      #env.render(state)
      action = agent.get_action(flatten(state), timestep)

      next_state, timestep = jax.jit(env.step)(state, action)
      reward = timestep.reward
      done = True if reward!=0 else False

      agent.memory.append((flatten(state), action, reward, flatten(next_state), done))
      agent.train()

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
