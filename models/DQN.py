import numpy as np
import tensorflow as tf
from collections import deque


class DeepQNetwork:
    """
    recall Q-Learning, how to update Q table
    old_Q = Q[s][a]
    Q[s][a] = old_Q + alpha * (r + gamma * np.max(Q[s_next]) - old_Q)

    notation in DQN
    Q_eval = net_eval(s), here is Q values with all actions
    Q_eval_wrt_a = Q[s][a] = Q_eval[a]

    Q_next = net_target(s_next) = Q[s_next]
    Q_target = r + gamma * np.max(Q_next)

    Loss = L2_Norm(Q_target - Q_eval_wrt_a) -> update Net_eval
    hard copy net_eval to net_target, e.g. every 300 iterations of Net_eval

    refs
    1. Udacity, Deep Reinforcement Learning
    2. MorvanZhou, DQN Tutorial
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 gamma=0.95,
                 learning_rate=0.01,
                 epsilon_start=1.0,
                 epsilon_decay=0.9999,
                 epsilon_min=0.01,
                 epsilon_fn=None,
                 memory_size=20000):

        self.n_actions = n_actions
        self.action_space = list(range(self.n_actions))
        self.n_features = n_features

        self.gamma = gamma  # score_log discount factor

        self.epsilon = epsilon_start  # greedy epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_epsilon = epsilon_fn if epsilon_fn is not None else self.decay_epislon

        self.learning_rate = learning_rate  # learning rate

        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)

        tf.reset_default_graph()

        self.batch_size = 50

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_target')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_eval')
        # define update net_target op
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # define network
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")
        self.a = tf.placeholder(tf.int32, [None, ], name="a")  # shape=(?,)
        self.r = tf.placeholder(tf.float32, [None, ], name="r")  # shape=(?,)
        self.s_next = tf.placeholder(tf.float32, [None, self.n_features], name="s_next")
        self.end_points = {}
        self.build_network()

        self.learn_step_counter = 0
        self.learn_upadte_steps = 300  # update net_target while net_eval updates every 300 times

        self.cost_list = []

        # must be the last line of init
        # must after graph definition
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        w_initializer = tf.keras.initializers.he_normal()
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope("net_eval") as scope:
            net_eval = tf.keras.layers.Dense(24,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="dense")(self.s)
            net_eval = tf.keras.layers.Dense(24,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="dense")(net_eval)
            net_eval = tf.keras.layers.Dense(self.n_actions,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="Q_eval")(net_eval)
            # Q_eval = net_eval(s) = old_Q
            self.end_points["Q_eval"] = net_eval  # [n_batch, n_actions]
            # note, Q_eval(shape=(?, 2)) has Q values with all actions

            # here we slice elements (indices=(n_actions, a)) from Q_eval and build a list wrt batch
            # suppose a has shape (n_batches,)
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # a_indices has shape (n_batch, 2) like [[0, action], [1, action], [2, action], ... ,[n_batches-1, action]]

            # Q_eval_wrt_a = Q[s][a] = Q_eval[a]
            self.end_points["Q_eval_wrt_a"] = tf.gather_nd(net_eval, indices=a_indices)  # shape=(None, )

        with tf.variable_scope("net_target") as scope:
            net_target = tf.keras.layers.Dense(24,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="dense")(self.s_next)
            net_target = tf.keras.layers.Dense(24,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="dense")(net_target)
            net_target = tf.keras.layers.Dense(self.n_actions,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="Q_next")(net_target)
            self.end_points["Q_next"] = net_target
            # Q_target = r + gamma * np.max(Q_next)
            net_target = self.r + self.gamma * tf.math.reduce_max(net_target, axis=1, name="Q_max")
            # Important! not calculating gradient from this op
            self.end_points["Q_target"] = tf.stop_gradient(net_target)

        with tf.variable_scope('loss'):
            self.end_points["loss"] = tf.reduce_mean(tf.squared_difference(self.end_points["Q_target"],
                                                                           self.end_points["Q_eval_wrt_a"],
                                                                           name='TD_error'))
        with tf.variable_scope('train'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                   momentum=0.9, name='Momentum',
                                                   use_nesterov=False)
            self.train_op = optimizer.minimize(self.end_points["loss"])

    def store_transition(self, s, a, r, s_next):
        # assert s,s_next -> np.array
        # assert a -> scalar
        # assert r -> scalar
        self.memory.append(np.concatenate([s, [a], [r], s_next])) # deque

    def choose_action(self, state):
        prob_choose = np.ones(self.n_actions) / self.n_actions * self.epsilon
        s = np.expand_dims(state, axis=0)
        action_value = self.sess.run(self.end_points["Q_eval"], feed_dict={self.s: s})
        # epsilon greedy
        prob_choose[np.argmax(action_value)] = 1 - (self.n_actions - 1.0)/self.n_actions * self.epsilon
        action = np.random.choice(self.action_space, p=prob_choose)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % 300 == 0:
            self.sess.run(self.target_replace_op)

        memory_size = len(self.memory)
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

        _, cost = self.sess.run([self.train_op, self.end_points["loss"]], feed_dict=feed_dict)
        self.cost_list.append(cost)
        self.update_epsilon()
        self.learn_step_counter += 1

    def decay_epislon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_list)), self.cost_list)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def interact(env, agent, num_episodes):
    global_step = 0

    for i_episode in range(num_episodes):
        # reset env for each episode
        state = env.reset()

        while True:

            action = agent.choose_action(state)
            state_next, reward, done, info = env.step(action)
            sars_sample = [state, action, reward, state_next]

            agent.store(sars_sample)

            if global_step > 200 and (global_step % 5 == 0):
                agent.learn()

            state = state_next
            if done is True:
                break

            global_step += 1


if __name__ == "__main__":
    import gym
    from tqdm import tqdm

    N_EPISODES = 100

    env = gym.make('CartPole-v0')
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Solved Requirements
    # Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    env = env.unwrapped  # important ?

    print(env.action_space)
    print(env.observation_space)
    # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = DeepQNetwork(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0])

    total_steps = 0

    smooth = 5
    smooth_score_log = []
    score_log = deque(maxlen=smooth)

    # pbar = tqdm(total=N_EPISODES)

    for i_episode in range(N_EPISODES):

        observation = env.reset()
        ep_r = 0
        steps = 0
        while True:
            steps += 1  # hold greater than 195 steps is a successful trial

            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_  # 细分开, 为了修改原配的 reward
            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # custom_reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

            RL.store_transition(observation, action, reward, observation_)

            # if total_steps > 1000:
            #     RL.learn()

            RL.learn()

            total_steps += 1
            observation = observation_

            if done:
                print('episode: ', i_episode,
                      'steps: ', steps,
                      'epsilon: ', round(RL.epsilon, 2))
                score_log.append(steps)
                smooth_score_log.append(np.mean(score_log))
                break

        # pbar.set_description('episode {}'.format(i_episode))
        # pbar.set_postfix(episode_reward=round(np.mean(score_log), 2))
        # pbar.update()

    # RL.plot_cost()
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(smooth_score_log)), smooth_score_log)
    plt.ylabel('avg episode reward')
    plt.xlabel('training steps')
    plt.show()