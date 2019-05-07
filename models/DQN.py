import numpy as np
import tensorflow as tf


class DeepQNetwork():
    """
    recall Q-Learning, how to update Q table
    old_Q = Q[s][a]
    Q[s][a] = old_Q + alpha * (r + gamma * np.max(Q[s_next]) - old_Q)

    in DQN, we separate this update into 2 networks, i.e. eval_network and target_network

    1. construct memory like [[S, A, R, S_next],[S, A, R, S_next],...]
    2. sample a memory batch, e.g. batch=1
    3. feed S to eval_network to calculate q_eval like Q[s][a] or old_Q in Q-learning
    4. feed S_next to target_network to calculate q_next like Q[s_next] in Q-learning
    5. q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
    6. loss=q_target - q_eval,
    7. consider q_target as labels, optimize eval_network to minimize loss

    """
    def __init__(self):
        self.n_features = None

    def build_network(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")  # input state


        # build eval network, Q_old


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
