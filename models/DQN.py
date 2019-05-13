import numpy as np
import tensorflow as tf
from _collections import deque


class DeepQNetwork():
    """
    recall Q-Learning, how to update Q table
    old_Q = Q[s][a]
    Q[s][a] = old_Q + alpha * (r + gamma * np.max(Q[s_next]) - old_Q)

    notation in DQN
    Q_eval = net_eval(s) = old_Q
    Q_next = net_target(s_next) = Q[s_next]
    Q_target = r + gamma * np.max(Q_next)

    Loss = L2_Norm(Q_target - Q_eval) -> update Net_eval
    hard copy net_eval to net_target, e.g. every 300 iterations of Net_eval

    refs
    1. Udacity, Deep Reinforcement Learning
    2. MorvanZhou, DQN Tutorial
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 gamma=0.9,
                 learning_rate=0.01,
                 epsilon_start=1.0,
                 epsilon_decay=0.9999,
                 epsilon_min=0.001,
                 epsilon_fn=None,
                 memory_size=2000):
        self.n_actions = n_actions
        self.action_space = list(range(self.n_actions))
        self.n_features = n_features

        self.gamma = gamma  # rewards discount factor

        self.epsilon = epsilon_start  # greedy epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_epsilon = epsilon_fn if epsilon_fn is not None else self.decay_epislon

        self.learning_rate = learning_rate  # learning rate

        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)

        self.batch_size = 32

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_target')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_eval')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.end_points = {}
        self.build_network()

        self.learn_step_counter = 0

        self.cost_list = []

        # must be the last line of init
        # must after graph definition
        self.sess = tf.Session()

    def build_network(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")
        self.a = tf.placeholder(tf.int32, [None, ], name="a")
        self.r = tf.placeholder(tf.float32, [None, ], name="r")
        self.s_next = tf.placeholder(tf.float32, [None, self.n_features], name="s_next")

        w_initializer = tf.keras.initializers.he_normal()
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope("net_eval") as scope:
            net_eval = tf.keras.layers.Dense(20,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="dense")(self.s)
            net_eval = tf.keras.layers.Dense(self.n_actions,
                                             activation=tf.keras.activations.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="Q_eval")(net_eval)
            self.end_points["Q_eval"] = net_eval  # [n_batch, n_actions]
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.end_points["Q_eval_wrt_a"] = tf.gather_nd(net_eval, indices=a_indices)  # shape=(None, )

        with tf.variable_scope("net_target") as scope:
            net_target = tf.keras.layers.Dense(20,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="dense")(self.s_next)
            net_target = tf.keras.layers.Dense(self.n_actions,
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=w_initializer,
                                               bias_initializer=b_initializer,
                                               name="Q_next")(net_target)
            self.end_points["Q_next"] = net_target
            net_target = self.r + self.gamma * tf.math.reduce_max(net_target, axis=1, name="Q_max")
            # Important! set not to calculate gradient from this op
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
        self.memory.append([s, a, r, s_next])  # deque

    def choose_action(self, state):
        prob_choose = np.ones(self.n_actions) / self.n_actions * self.epsilon
        action_value = self.sess.run(self.end_points["Q_eval"], feed_dict={self.s: state})
        prob_choose[np.argmax(action_value)] = 1 - (self.nA - 1.0)/self.nA * self.epsilon
        action = np.random.choice(self.action_space, p=prob_choose)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % 300 == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        memory_size = len(self.memory)
        sample_index = np.random.choice(range(memory_size), size=self.batch_size)  # replace = False
        batch_memory = [self.memory[i] for i in sample_index]

        s, a, r, s_next = zip(*batch_memory)
        feed_dict = {self.s: np.array(s, dtype=np.float),
                     self.a: np.array(a, dtype=np.int32),
                     self.r: np.array(r, dtype=np.float32),
                     self.s_next: np.array(s_next, dtype=np.float)}

        _, cost = self.sess.run([self.train_op, self.end_points["loss"]], feed_dict=feed_dict)
        self.cost_list.append(cost)
        self.update_epsilon()

    def decay_epislon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


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

    env = gym.make('CartPole-v0')  # 定义使用 gym 库中的那一个环境
    env = env.unwrapped  # 不做这个会有很多限制

    print(env.action_space)  # 查看这个环境中可用的 action 有多少个
    print(env.observation_space)  # 查看这个环境中可用的 state 的 observation 有多少个
    print(env.observation_space.high)  # 查看 observation 最高取值
    print(env.observation_space.low)  # 查看 observation 最低取值

    # 定义使用 DQN 的算法
    RL = DeepQNetwork(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0])

    total_steps = 0  # 记录步数

    for i_episode in range(100):

        # 获取回合 i_episode 第一个 observation
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()  # 刷新环境

            action = RL.choose_action(observation)  # 选行为

            observation_, reward, done, info = env.step(action)  # 获取下一个 state

            x, x_dot, theta, theta_dot = observation_  # 细分开, 为了修改原配的 reward
            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

            # 保存这一组记忆
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > 1000:
                RL.learn()  # 学习

            ep_r += reward
            if done:
                print('episode: ', i_episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                break

            observation = observation_
            total_steps += 1
    # 最后输出 cost 曲线
    #RL.plot_cost()