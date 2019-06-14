import numpy as np
import tensorflow as tf
from collections import deque
import time


class DeepQNetwork:
    """
    recall Q-Learning, how to update Q table
    Q[s][a] = Q[s][a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])

    notation in DQN, i'd like to call eval_net as "policy net"
    Q_eval = eval_net(s), here is Q values with all actions
    Q_eval_wrt_a = Q[s][a] = Q_eval[a]

    Q_next = target_net(s_next) = Q[s_next]
    Q_target = r + gamma * np.max(Q_next)

    due to contraction mapping of Bellman operator
    min L2_Norm(Q_target - Q_policy) to find the fix point of
    Q* = BQ*
    so, define
    Loss = L2_Norm(Q_target - Q_eval_wrt_a) -> update eval_net
    hard copy eval_net to target_net, e.g. every 300 iterations of eval_net

    refs
    1. Udacity, Deep Reinforcement Learning Nanodegree
    2. Udacity, ud600 Reinforcement Learning
    2. DQN Tutorial by MorvanZhou
    3. Cartpole - Introduction to Reinforcement Learning (DQN - Deep Q-Learning) by Greg Surma
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 gamma=0.99,
                 learning_rate=0.0005,
                 update_every=4,
                 batch_size=64,
                 memory_size=int(1e5),
                 model_save_path="./models/",
                 log_save_path="./logs/",
                 output_graph=True):

        # ------------------------------------------
        # model parameters
        # ------------------------------------------
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate

        self.gamma = gamma  # reward discounter

        self.update_every = update_every

        self.memory_size = memory_size
        self.batch_size = batch_size

        # initialize zero memory [s, a, r, s_]
        self.memory = deque(maxlen=self.memory_size)

        # total learning step
        self.learn_step_counter = 0

        # ------------------------------------------
        # define network
        # ------------------------------------------
        # definition sequence in tf is important, so i put all definition together in __init__,
        # in case of stupid mistakes
        self.endpoints = {}
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_next = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope("eval_net") as scope:
            eval_net = tf.keras.layers.Dense(64,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="dense1")(self.s)
            eval_net = tf.keras.layers.Dense(64,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="dense2")(eval_net)
            eval_net = tf.keras.layers.Dense(self.n_actions,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="Q_eval")(eval_net)
            self.endpoints["Q_eval"] = eval_net  # [n_batch, n_actions]
            # note, Q_eval(shape=(?, 2)) has Q values with all actions

            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # a_indices has shape (n_batch, 2) like [[0, action], [1, action], [2, action], ... ,[n_batches-1, action]]
            self.endpoints["Q_eval_wrt_a"] = tf.gather_nd(self.endpoints["Q_eval"], indices=a_indices)  # shape=(None, )

        with tf.variable_scope("target_net") as scope:
            target_net = tf.keras.layers.Dense(64,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="dense1")(self.s_next)
            target_net = tf.keras.layers.Dense(64,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="dense2")(target_net)
            target_net = tf.keras.layers.Dense(self.n_actions,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="Q_next")(target_net)
            self.endpoints["Q_next"] = target_net
            # Q_target = r + gamma * np.max(Q_next)
            q_target = self.r + self.gamma * tf.math.reduce_max(self.endpoints["Q_next"], axis=1, name="Q_max")
            # Important! not calculating gradient FROM this op to input node
            self.endpoints["Q_target"] = tf.stop_gradient(q_target)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.endpoints["Q_target"],
                                                             self.endpoints["Q_eval_wrt_a"], name='TD_error'))
        with tf.variable_scope('train'):
            # self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss,
                                                                                   global_step=self.global_step)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # ToDo, soft-update with tau ?
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # log and saver
        # cao, saver must defined after network!!!
        self.saver = tf.train.Saver(max_to_keep=1)
        self.saved_path = model_save_path

        # no op definition after session started
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if output_graph is True:
            file_writer = tf.summary.FileWriter(logdir=log_save_path, graph=self.sess.graph)

    def act(self, state, eps):
        s = np.expand_dims(state, axis=0)
        if np.random.uniform() < eps:
            action = np.random.randint(0, self.n_actions)
        else:
            # forward feed the state and get q value for every actions
            actions_value = self.sess.run(self.endpoints["Q_eval"], feed_dict={self.s: s})
            action = np.argmax(actions_value)
        return action

    def step(self, s, a, r, s_next, done):
        # assert s,s_next -> np.array
        # assert a -> scalar
        # assert r -> scalar
        # assert done -> scalar
        self.memory.append(np.concatenate([s, [a], [r], s_next, [done]])) # deque
        memory_size = len(self.memory)
        if memory_size < self.batch_size:
            return

        # check to replace target parameters
        # in cartpole-v0 experiment, if replace too frequent,
        # the convergence will be not stable. sometime converge, sometime not
        if self.learn_step_counter % 100 == 0:
            self.sess.run(self.target_replace_op)

        sample_index = np.random.choice(range(memory_size), size=self.batch_size)  # replace = False
        batch_memory = np.array([self.memory[i] for i in sample_index])

        s_sample = batch_memory[:, 0:self.n_features]                              # shape=(#batch, n_features)
        a_sample = batch_memory[:, self.n_features:self.n_features+1].squeeze()    # shape=(#batch,)
        r_sample = batch_memory[:, self.n_features+1:self.n_features+2].squeeze()  # shape=(#batch,)
        s_next_sample = batch_memory[:, self.n_features+2:self.n_features*2+2]     # shape=(#batch, n_features)
        done_sample = batch_memory[:, self.n_features*2+2:self.n_features*2+3]     # shape=(#batch,)

        feed_dict = {self.s: s_sample.astype(np.float32),
                     self.a: a_sample.astype(np.int32),
                     self.r: r_sample.astype(np.float32),
                     self.s_next: s_next_sample.astype(np.float32)}

        _, loss, self.learn_step_counter = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)

    def save(self):
        self.saver.save(self.sess, self.saved_path + 'dqn.ckpt', global_step=self.global_step)

    def restore(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.saved_path))
        self.learn_step_counter = self.sess.run(self.global_step)


def debug_print(sess):
    # default graph
    vars = tf.trainable_variables()
    vars_vals = sess.run(vars)
    for var, val in zip(vars, vars_vals):
        print(var.name)
        print("shape: {}, sum: {}".format(val.shape, np.sum(val)))


if __name__ == "__main__":
    import gym
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or any '0,1'

    env = gym.make('CartPole-v0')
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Solved Requirements
    # Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    agent = DeepQNetwork(n_actions=env.action_space.n,
                         n_features=env.observation_space.shape[0],
                         learning_rate=0.01)
    # debug_print(sess=agent.sess)
    # agent.restore()
    # print("----------")
    # debug_print(sess=agent.sess)

    N_EPISODES = 100
    scores_log = []

    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start

    for i_episode in range(N_EPISODES):

        state = env.reset()
        steps = 0
        while True:
            #env.render()
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = next_state
            # smaller reward, when not in the central
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # smaller reward, when not perpendicular
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            # # if not use customized reward, the last reward +1 with done==True should be considered
            # # as an end game action, like a penalty.
            # reward = -1*reward if done is True else reward

            agent.step(state, action, reward, next_state, done)

            steps += 1
            state = next_state
            eps = max(eps_end, eps_decay * eps)
            if done:
                break
            if steps > 500:
                # sucessful episode
                break

        scores_log.append(steps)
        print("----------")
        print("episode {}, epsilon {}, learn_steps {}, steps {}".format(i_episode,
                                                                        round(eps, 2),
                                                                        agent.learn_step_counter,
                                                                        steps))
        debug_print(sess=agent.sess)

    agent.save()