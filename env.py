import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from IPython.display import clear_output

def run(config, agents):

  scores = []
  scores_avg = deque(maxlen=100)

  episodes = config.get('episodes', 500)
  train_mode = config.get('train_mode', True)
  add_noise = config.get('add_noise', train_mode)
  noise_scale = config.get('noise_scale', 1.0)
  noise_decay = config.get('noise_decay', 0.99)
  plot_every = config.get('plot_every', 10)
  max_timestep = config.get('max_timestep', 1000)
  label = config.get('label', 'Agent Score')
  
  env = config.get('env')
  brain_name = env.brain_names[0]

  for i in range(1, episodes+1):

    env_info = env.reset(train_mode=train_mode)[brain_name]
    num_agents = len(env_info.agents)
    for agent in agents:
      agent.reset()

    states = env_info.vector_observations
    tmp_scores = np.zeros(num_agents)
    t = 0

    while True:
      actions = [
        agent.act(state, add_noise=add_noise, noise_scale=noise_scale)
        for agent, state in zip(agents, states)
      ]
      env_info = env.step(np.clip(actions, -1, 1))[brain_name]
      next_states = env_info.vector_observations
      rewards = env_info.rewards
      dones = env_info.local_done
      tmp_scores += env_info.rewards

      if train_mode:
        for n, agent in enumerate(agents):
          agent.step(states[n], actions[n], rewards[n], next_states[n], dones[n])

      states = next_states

      t += 1
      if np.any(dones) or max_timestep and t == max_timestep:
        score = np.max(tmp_scores)
        scores_avg.append(score)
        avg = np.mean(scores_avg)
        scores.append((score, avg))
        break

    noise_scale *= noise_decay

    if i % plot_every == 0:
      clear_output(True)

      _, ax1 = plt.subplots(1, 1, figsize=(18, 9))
      values, avg_values = zip(*scores)
      x = range(len(values))
      ax1.set_title(label)
      ax1.plot(x, values, label='Score', color='lightblue')
      ax1.plot(x, avg_values, label="Average over 100 episodes", color='blue')
      ax1.legend()
      plt.show()

    if train_mode and avg >= 0.5:
      print("Solved in {} episodes!".format(i))
      break