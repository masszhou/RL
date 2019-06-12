import numpy as np
import tensorflow as tf
from collections import deque


class DeepQNetwork:
    """
    recall Q-Learning, how to update Q table
    Q[s][a] = Q[s][a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])

    notation in DQN
    Q_eval = eval_net(s), here is Q values with all actions
    Q_eval_wrt_a = Q[s][a] = Q_eval[a]

    Q_next = target_net(s_next) = Q[s_next]
    Q_target = r + gamma * np.max(Q_next)

    due to contraction mapping of Bellman operator
    min L2_Norm(Q_target - Q_eval_wrt_a) to find the fix point of
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
                 gamma=0.95,
                 learning_rate=0.01,
                 epsilon_start=1.0,
                 epsilon_decay=0.999,
                 epsilon_min=0.05,
                 replace_target_iter=100,
                 batch_size=32,
                 memory_size=20000,
                 output_graph=True):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate

        self.gamma = gamma  # reward decay

        self.replace_target_iter = replace_target_iter

        self.memory_size = memory_size
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = deque(maxlen=self.memory_size)

        # consist of [target_net, evaluate_net]
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_next = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.endpoints = {}
        self.build_network()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_log = []

    def build_network(self):
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope("eval_net") as scope:
            eval_net = tf.keras.layers.Dense(20,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="dense")(self.s)
            eval_net = tf.keras.layers.Dense(self.n_actions,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="Q_eval")(eval_net)
            self.endpoints["Q_eval"] = eval_net # [n_batch, n_actions]
            # note, Q_eval(shape=(?, 2)) has Q values with all actions

            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # a_indices has shape (n_batch, 2) like [[0, action], [1, action], [2, action], ... ,[n_batches-1, action]]
            self.endpoints["Q_eval_wrt_a"] = tf.gather_nd(self.endpoints["Q_eval"], indices=a_indices)  # shape=(None, )

        with tf.variable_scope("target_net") as scope:
            target_net = tf.keras.layers.Dense(20,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="dense")(self.s_next)
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
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_next):
        # assert s,s_next -> np.array
        # assert a -> scalar
        # assert r -> scalar
        self.memory.append(np.concatenate([s, [a], [r], s_next])) # deque

    def choose_action(self, state):
        s = np.expand_dims(state, axis=0)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            # forward feed the state and get q value for every actions
            actions_value = self.sess.run(self.endpoints["Q_eval"], feed_dict={self.s: s})
            action = np.argmax(actions_value)
        return action

    def learn(self):
        memory_size = len(self.memory)
        if memory_size < self.batch_size:
            return

        # check to replace target parameters
        # if replace too frequent, the convergence will be not stable. sometime converge, sometime not
        if self.learn_step_counter % 100 == 0:
            self.sess.run(self.target_replace_op)

        sample_index = np.random.choice(range(memory_size), size=self.batch_size)  # replace = False
        batch_memory = np.array([self.memory[i] for i in sample_index])

        s = batch_memory[:, 0:self.n_features]  # shape=(#batch, 4)
        a = batch_memory[:, self.n_features:self.n_features+1].squeeze()  # shape=(#batch,)
        r = batch_memory[:, self.n_features+1:self.n_features+2].squeeze()  # shape=(#batch,)
        s_next = batch_memory[:, self.n_features+2:]  # shape=(#batch,4)

        feed_dict = {self.s: s.astype(np.float32),
                     self.a: a.astype(np.int32),
                     self.r: r.astype(np.float32),
                     self.s_next: s_next.astype(np.float32)}

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        self.cost_log.append(loss)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_log)), self.cost_log)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == "__main__":
    from tqdm import tqdm
    import gym

    env = gym.make('CartPole-v0')
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Solved Requirements
    # Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = DeepQNetwork(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      learning_rate=0.01)

    N_EPISODES = 100
    pbar = tqdm(total=N_EPISODES)
    for i_episode in range(N_EPISODES):

        state = env.reset()
        steps = 0
        while True:
            steps += 1
            env.render()

            action = RL.choose_action(state)

            state_next, reward, done, info = env.step(action)

            # # the smaller theta and closer to center the better
            # x, x_dot, theta, theta_dot = state_next
            # # smaller reward, when not in the central
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # # smaller reward, when not perpendicular
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # reward = r1 + r2

            # if not use customized reward, the last reward +1 with done==True should be considered
            # as an end game action, like a penalty.
            reward = -1*reward if done is True else reward

            RL.store_transition(state, action, reward, state_next)
            RL.learn()

            if done:
                break
            state = state_next

        pbar.set_description('episode {}, epsilon {}'.format(i_episode, round(RL.epsilon, 2)))
        pbar.set_postfix(steps=steps)
        pbar.update()
    pbar.close()