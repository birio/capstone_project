
def clean_str_config(str_config):
   str_config = str_config.replace("[","")
   str_config = str_config.replace("]","")
   str_config = str_config.replace(" ","")
   str_config = str_config.replace(",","_")
   return str_config


def print_episode(i_episode, reward, n_comb, n_states, str_config):
   if i_episode%50==0:
      print("episode ", i_episode, ": reward: ", reward, "\t(", int(reward*(n_comb+n_states)), "/", (n_comb+n_states), ")")
      log_config = clean_str_config(str_config) + '.log'
      if i_episode==0:
         output_f = open(log_config, "w")
      else:
         output_f = open(log_config, "a")
      output_f.write("episode %s"   % repr(i_episode)                )
      output_f.write(": reward: %s" % repr(reward)                   )
      output_f.write("\t (%s"       % repr(reward*(n_comb+n_states)) )
      output_f.write("/%s)\n"       % repr(n_comb+n_states)          )
      output_f.closed

def print_stats(dut, n_states, best_reward, best_episode_states):
   stats_f = open("stats.log", "w")
   stats_f.write("dut.states_covered = %s\n" % repr(dut.states_covered) )

   for key, items in dut.comb_covered.items():
      stats_f.write("dut.comb_covered[%s] "   % repr(key)     )
      stats_f.write("%s\t"                    % repr(items)   )
   stats_f.write("\n")

   stats_f.write("dut.COMB = %s"             % repr(dut.COMB) )
   stats_f.write("dut.DUT  = %s"             % repr(dut.DUT ) )
   stats_f.closed

   print("best reward = ", best_reward)
   # best_episode_f = open("best_epsiode.log", "w")
   # temp=[]
   # for i in range (len(best_episode_states)-1):
   #    temp.append(best_episode_states[i])
   #    if(best_episode_states[i+1] != best_episode_states[i]) or ((i+1) == len(best_episode_states)):
   #       best_episode_f.write("%s" % repr(temp[0])   )
   #       best_episode_f.write("("                    )
   #       best_episode_f.write("%s" % repr(len(temp)) )
   #       best_episode_f.write(")\t"                  )
   #       temp=[]
   # best_episode_f.closed

