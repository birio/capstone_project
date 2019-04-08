from Dut import Dut
import matplotlib.pyplot as plt
import numpy as np
import pdb
import print_stats

CONFIGS = [ [False,   8,   32],
            [ True,   8,   32],
            [False,  32,  256],
            [ True,  32,  256],
            [False, 128, 1024],
            [ True, 128, 1024] ]

for config in CONFIGS:

   print(config)
   str_config = 'random_' + str(config)
   DO_MERGE = config[0] 
   N_INPUTS = config[1] 
   N_STATES = config[2] 
  
 
   num_episodes = 1000
   t_max        =  500
   
   rewards_l = []
   
   best_reward = -1
   best_episode_states = []
   episode_states = []
   
   
   dut = Dut(N_STATES, N_INPUTS)
   
   for i_episode in range(1, num_episodes+1):
       n_comb = dut.n_comb
       state = dut.reset(DO_MERGE)
       step = 1
       done = False
       while step <= t_max and not done:
           episode_states.append(state)
           # apply random stimuli
           action = int((N_INPUTS)*np.random.random())
           next_state, reward, done = dut.step(state, action)
           state = next_state
           step = step + 1
           if done or (step == t_max):
              print_stats.print_episode(i_episode, reward, n_comb, N_STATES, str_config)
              rewards_l.append(reward)
              if reward > best_reward:
                 best_reward = reward
                 best_episode_states = episode_states
   
   print_stats.print_stats(dut, N_STATES, best_reward, best_episode_states)
   plt.plot(rewards_l)
   png_config = 'rewards_' + print_stats.clean_str_config(str_config) + '.png'
   plt.savefig(png_config)


# stat on action generated per state
