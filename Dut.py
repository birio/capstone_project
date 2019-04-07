import numpy as np
import pdb


class Dut:

   def compute_reward_states(self, states_covered):
       reward_states = sum(states_covered)
       return reward_states

   def compute_reward_comb(self, comb_covered):
       reward_comb = 0
       for i in range (self.n_states):
          for j in range (self.n_states):
             if(self.comb_covered[i][j] == 1):
                reward_comb = reward_comb + 1
       return reward_comb

   def compute_reward(self, states_covered, comb_covered):
       reward = (self.compute_reward_states(states_covered) + self.compute_reward_comb(comb_covered))/(self.n_comb + self.n_states)
       return reward

   def add_adj_list(self, max_adj_states, state):
       adj_states = np.random.random_integers(max_adj_states)
       rnd_states_list = np.random.choice(self.n_states, adj_states, replace=False)
       if state in self.DUT and self.DUT[state] not in rnd_states_list:
          rnd_states_list = np.append(rnd_states_list, self.DUT[state])
       self.DUT.update({state: rnd_states_list})
       self.all_states_s.add(state)
       self.all_states_s.update(rnd_states_list) 

   def __init__(self, n_states, n_inputs):
      seed = 42
      np.random.seed(seed)
      max_adj_states = np.random.random_integers(n_states) # max number of adjacent states for each state
      
      self.DUT = {} # dictioary adjacent list
      self.COMB = [[None for i in range(n_states)] for j in range(n_states)]
      self.states_covered = [0 for i in range(n_states)]
      self.comb_covered   = [["-" for i in range(n_states)] for j in range(n_states)]
      self.n_comb = 0
      self.n_states = n_states
      self.all_states_s = set()

      s = 0
      self.add_adj_list(max_adj_states, s)
      for s in range(1, n_states):
         if s not in self.all_states_s:
            rnd_state = np.random.choice(list(self.all_states_s))
            self.DUT.update({rnd_state : [s]})
         self.add_adj_list(max_adj_states, s)
      
      for key, items in self.DUT.items():
         for i in range(n_states):
            if i in items:
               self.COMB[key][i]=np.random.random_integers(n_inputs)
               self.comb_covered[key][i] = 0
               self.n_comb = self.n_comb + 1

      assert len(self.all_states_s) == self.n_states , "len(all_states_s) is " % len(self.all_states_s)

      # self.classes_f = open("classes.dat", "w")
      # with open("classes.dat") as classes_f:
      #    self.classes_f.write("n_states = %s\n" % repr(self.n_states))
      #    self.classes_f.write("n_comb   = %s\n" % repr(self.n_comb))
      #    self.classes_f.write(repr(self.DUT ))
      #    self.classes_f.write("\n")
      #    self.classes_f.write(repr(self.COMB))
      # self.classes_f.closed
      

   # next_state, reward, done = dut.step(state, action)
   def step(self, state, action):
      state_array = self.DUT[state]
      next_state = state
      for i in state_array:
         if self.COMB[state][i]==action:
            self.states_covered[i]=1
            self.comb_covered[state][i]=1
            next_state = i
            break
      reward = self.compute_reward(self.states_covered, self.comb_covered)
      done   = (reward == 1)
      return (next_state, reward, done)

   def reset(self, do_merge):
      if not do_merge:
         self.states_covered = [0 for i in range(self.n_states)]
         self.comb_covered   = [["-" for i in range(self.n_states)] for j in range(self.n_states)]
         for i in range (self.n_states):
            for j in range (self.n_states):
               if(self.COMB[i][j] != None):
                  self.comb_covered[i][j] = 0
      return 0;
   
