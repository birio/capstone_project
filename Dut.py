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
      self.prev_coverage = 0

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

      print("DUT[0]: ", self.DUT[0])
      for i in self.DUT[0]:
         print("COMB[(0, ", i, ")] = ", self.COMB[(0, i)])

      print("DUT[30]: ", self.DUT[30])
      for i in self.DUT[30]:
         print("COMB[(30, ", i, ")] = ", self.COMB[(30, i)])

      print("DUT[15]: ", self.DUT[15])
      for i in self.DUT[15]:
         print("COMB[(15, ", i, ")] = ", self.COMB[(15, i)])

      print("DUT[26]: ", self.DUT[26])
      for i in self.DUT[26]:
         print("COMB[(26, ", i, ")] = ", self.COMB[(26, i)])

      print("DUT[17]: ", self.DUT[17])
      for i in self.DUT[17]:
         print("COMB[(17, ", i, ")] = ", self.COMB[(17, i)])


      # self.classes_f = open("classes.dat", "w")
      # with open("classes.dat") as classes_f:
      #    self.classes_f.write("n_states = %s\n" % repr(self.n_states))
      #    self.classes_f.write("n_comb   = %s\n" % repr(self.n_comb))
      #    self.classes_f.write(repr(self.DUT ))
      #    self.classes_f.write("\n")
      #    self.classes_f.write(repr(self.COMB))
      # self.classes_f.closed


   # next_state, reward, done = dut.step(state, action)

   # TODO bonus for consecutive non negative reward

   def step(self, state, action):
      state_array = self.DUT[state]
      next_state = state
      reward = -1
      self.prev_coverage = self.coverage
      for i in state_array:
         if self.COMB[(state, i)] == action:
            self.states_covered[i]=1
            # reward = 0.1
            reward = 1
            if (self.comb_covered[(state, i)] == 0):
               self.comb_covered.update({(state, i): 1})
               self.coverage = self.compute_reward(self.states_covered, self.comb_covered)
               all_comb = True
               for k in state_array:
                  if (self.comb_covered[(state, k)] == 0):
                     all_comb = False
               if all_comb:
                  reward = 5
            next_state = i
            break

      if self.coverage > 0.5 and self.prev_coverage < 0.5:
         reward = reward + 5
      elif self.coverage > 0.6 and self.prev_coverage < 0.6:
         reward = reward + 10
      elif self.coverage > 0.7 and self.prev_coverage < 0.7:
         reward = reward + 100
      elif self.coverage > 0.75 and self.prev_coverage < 0.75:
         reward = reward + 500
      elif self.coverage > 0.8 and self.prev_coverage < 0.8:
         reward = reward + 2000
      elif self.coverage > 0.85 and self.prev_coverage < 0.85:
         reward = reward + 3000
      elif self.coverage > 0.9 and self.prev_coverage < 0.9:
         reward = reward + 5000
      elif self.coverage > 0.95:
         reward = reward * reward

      done          = (self.coverage == 1)

      return (next_state, reward, done)

   def reset(self, do_merge):
      self.coverage = 0
      if not do_merge:
         self.states_covered = [0 for i in range(self.n_states)]
         for key, items in self.comb_covered.items():
            self.comb_covered.update({key: 0})
      return 0;
   
