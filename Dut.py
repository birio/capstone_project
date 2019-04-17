import numpy as np
import pdb


class Dut:

   def compute_reward_states(self, states_covered):
       reward_states = sum(states_covered)
       return reward_states

   def compute_reward_comb(self, comb_covered):
       reward_comb = 0
       for key, items in self.comb_covered.items():
          reward_comb = reward_comb + items
       return reward_comb

   def compute_reward(self, states_covered, comb_covered):
       coverage = (self.compute_reward_states(states_covered) + self.compute_reward_comb(comb_covered))/(self.n_comb + self.n_states)
       return coverage

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
      max_adj_states = np.random.random_integers(min(n_inputs, n_states)-1) # max number of adjacent states for each state
      
      self.DUT = {} # dictioary adjacent list
      self.COMB = {} # dictionary of combinatory path between states
      self.states_covered = [0 for i in range(n_states)]
      self.comb_covered   = {}
      self.n_comb = 0
      self.n_states = n_states
      self.n_inputs = n_inputs
      self.all_states_s = set()
      self.coverage = 0

      s = 0
      self.add_adj_list(max_adj_states, s)
      for s in range(1, n_states):
         if s not in self.all_states_s:
            rnd_state = np.random.choice(list(self.all_states_s))
            rnd_state_items = [s]
            if rnd_state in self.DUT.keys():
               rnd_state_items = np.unique(np.append(rnd_state_items, self.DUT[rnd_state]))
            self.DUT.update({rnd_state : rnd_state_items})
         self.add_adj_list(max_adj_states, s)
     
      for key, items in self.DUT.items():
         rnd_actions_l = np.random.choice(n_inputs, len(self.DUT[key]), replace=False)
         for i in range(len(items)):
            self.COMB.update({ (key, items[i]) : rnd_actions_l[i] })
            self.comb_covered.update({ (key, items[i]) : 0 })
            self.n_comb = self.n_comb + 1

      assert len(self.all_states_s) == self.n_states , "len(all_states_s) is " % len(self.all_states_s)

      self.tot_coverage = len(self.states_covered)
      for key, items in self.comb_covered.items():
          self.tot_coverage = self.tot_coverage + 1
      print("tot_coverage = ", self.tot_coverage)


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
      reward = -0.1
      for i in state_array:
         if self.COMB[(state, i)] == action:
            self.states_covered[i]=1
            reward = 0
            if (self.comb_covered[(state, i)] == 0):
               self.comb_covered.update({(state, i): 1})
               self.coverage = self.compute_reward(self.states_covered, self.comb_covered)
               reward = self.coverage
            next_state = i
            break
      done          = (self.coverage == 1)
      return (next_state, reward, done)

   def reset(self, do_merge):
      self.coverage = 0
      if not do_merge:
         self.states_covered = [0 for i in range(self.n_states)]
         for key, items in self.comb_covered.items():
            self.comb_covered.update({key: 0})
      return 0;
   
