from Dut import Dut
import matplotlib.pyplot as plt
import pdb
import print_stats

import tensorflow as tf
import numpy as np
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        out = tflearn.fully_connected(net, 300)
        out = tflearn.layers.normalization.batch_normalization(out)
        out = tflearn.activations.relu(out)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(
        #     net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to 0 to action_bound-1
        scaled_out = tflearn.fully_connected(out, self.action_bound, activation="softmax")
        # scaled_out = ((self.action_bound-1)/2) + tf.multiply(out, (self.action_bound-1)/2)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, dut, args, actor, critic, actor_noise, do_merge, n_inputs, n_states, k_nearest_neighbors):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        counter = np.zeros(n_inputs)
        rew     = np.zeros(n_inputs)
        n_comb = dut.n_comb
        s = dut.reset(do_merge)

        ep_reward = 0
        ep_ave_max_q = 0

        best_episode_states = []
        episode_states = []

        for j in range(int(args['max_episode_len'])):
            episode_states.append(s)

            # get the proto_action's k nearest neighbors
            # actions = self.action_space.search_point(proto_action, k_nearest_neighbors)[0] # TODO efficient knn : closest in value, or in probability to being chosen?


            # TODO: print randomness ratio
            # REVISIT is it ok?
            # rnd_action = np.random.choice(np.arange(n_inputs))
            # eps = 1./(1+(float(args['max_episode_len'])*i + j)/250000)
            # if (np.random.random() < eps):
            #    proto_action = np.array([[rnd_action]])
            # else:
            #    proto_action = np.array([[np.argmax(actor.predict( np.reshape(s, (1, actor.s_dim))) )]])

            probs = actor.predict( np.reshape(s, (1, actor.s_dim)))
            probs = np.reshape(probs, (n_inputs, ))
            proto_action = np.random.choice(n_inputs, p = probs ) 

            a = proto_action

            # actions = np.round(proto_action)
            # # self.data_fetch.set_ndn_action(actions[0].tolist()) # todo
            # # make all the state, action pairs for the critic
            # states = np.tile(s, [len(actions), 1])
            # # evaluate each pair through the critic
            # actions_evaluation = critic.predict_target(states, actions)
            # # find the index of the pair with the maximum value
            # max_index = np.argmax(actions_evaluation)
            # a = actions[max_index]

            s2, r, terminal = dut.step(s, a)

            #         def add(self,                       s,                             a, r,        t,                             s2):
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
            #   s_batch, a_batch, r_batch, t_batch, s2_batch
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, np.array([[np.argmax(actor.predict_target(s2_batch))]]) ) #REVISIT

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, np.array([[np.argmax(a_outs)]]))
                actor.train(s_batch, grads[0])
                # pdb.set_trace()

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

                for x in range(0, len(s_batch)):
                   if(s_batch[x]==0):
                      y = a_batch[x]
                      counter[y] = counter[y] + 1 
                      rew[y] = rew[y] + r_batch[x]

               

            s = s2
            ep_reward += r

            if ( terminal or (j == (int(args['max_episode_len'])-1)) ):

                # REVISIT
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print ("pred[0] = ", actor.predict( np.reshape(0, (1, actor.s_dim))))
                print(counter)
                print(rew)
                print('| Coverage: {:.4f} ({:d}/{:d}) | Reward: {:.1f} | Episode: {:d} | Qmax: {:.4f}'.format(dut.coverage, int(dut.coverage*dut.tot_coverage), dut.tot_coverage, ep_reward, \
                        i, (ep_ave_max_q / float(j))))
                break

# TODO whats the meaning of Qmax?

def main(args):

    with tf.Session() as sess:

        # TODO: for configs
        DO_MERGE = False
        N_INPUTS = 8
        N_STATES = 32
        k_ratio  = 0.1 # REVISIT

        k_nearest_neighbors = max(1, int(N_INPUTS * k_ratio))

        dut = Dut(N_STATES, N_INPUTS)
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))

        state_dim = 1
        action_dim = 1
        action_bound = N_INPUTS
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        train(sess, dut, args, actor, critic, actor_noise, DO_MERGE, N_INPUTS, N_STATES, k_nearest_neighbors)

    print(dut.states_covered)
    print(dut.comb_covered)
    print(dut.DUT)
    print(dut.COMB)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=42)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=500)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)

    # TODO prints and statsa
 
    # TODO Q    values are initialized to zero
    # TODO put coverage in NN
    # TODO RNN
    # TODO print actions variance per episode
    # TODO reward: hidden state
    # TODO reward: big end reward
    # TODO what if Qmax explose?
    # TODO gamma=1
    # TODO random with minarg with threshold
    # TODO continuous tasks
    # TODO train with balanced actions
