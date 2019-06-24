# author: Zhiliang Zhou
# E-Mail: zhouzhiliang@gmail.com
# n-bar linkage robot arm env
# learned from 2-bar linkage example, https://github.com/MorvanZhou/train-robot-arm-from-scratch

import numpy as np
import pyglet


def cbox_to_bbox(cbox):
    """
    convert center box definition to bounding box definition
    :param cbox: (x_center, y_center, width, height
    :return: bbox: (x_topleft, y_topleft, x_bottomright, y_bottomright)
    """
    bbox = np.zeros(4)
    bbox[0] = cbox[0] - cbox[2] / 2
    bbox[1] = cbox[1] - cbox[3] / 2
    bbox[2] = cbox[0] + cbox[2] / 2
    bbox[3] = cbox[1] + cbox[3] / 2
    return bbox


def bbox_to_cbox(bbox):
    cbox = np.zeros(4)
    cbox[0] = (bbox[0] + bbox[2]) / 2
    cbox[1] = (bbox[1] + bbox[3]) / 2
    cbox[2] = bbox[2] - bbox[0]
    cbox[3] = bbox[3] - bbox[1]
    return cbox


class Viewer(pyglet.window.Window):

    def __init__(self,
                 goal_bbox,
                 n_bars=2,
                 bar_width=6,
                 canvas_width=400,
                 canvas_height=400,
                 arm_center_x=200,
                 arm_center_y=200):
        super(Viewer, self).__init__(width=canvas_width, height=canvas_height, resizable=False, caption='Arm', vsync=False)
        # window background color
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.bar_width = bar_width
        self.n_bars = n_bars
        self.center_coord = np.array([arm_center_x, arm_center_y])
        self.bar_position = np.zeros([n_bars, 4])

        # display whole batch at once
        self.batch = pyglet.graphics.Batch()
        # add target box
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal_bbox[0], goal_bbox[1],
                     goal_bbox[0], goal_bbox[3],
                     goal_bbox[2], goal_bbox[3],
                     goal_bbox[2], goal_bbox[1]]),
            ('c3B', (86, 109, 249) * 4))    # color

        # add bars
        self.bars = []
        for i in range(n_bars):
            self.bars.append(self.batch.add(4, pyglet.gl.GL_QUADS, None,    # 4 corners
                                            ('v2f', [250, 250,              # x1, y1
                                                     250, 300,              # x2, y2
                                                     260, 300,              # x3, y3
                                                     260, 250]),            # x4, y4
                                            ('c3B', (249, 86, 86) * 4,)))    # color

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def update(self, bar_state):
        """
        :param bar_state: (x1,y1,x2,y2,r,theta)
        :return:
        """
        n_bars = bar_state.shape[0]
        for id in range(n_bars):
            p1x = bar_state[id][0] + 0.5 * self.bar_width * np.cos(np.pi * 0.5 - bar_state[id][4])
            p1y = bar_state[id][1] - 0.5 * self.bar_width * np.sin(np.pi * 0.5 - bar_state[id][4])

            p2x = bar_state[id][0] - 0.5 * self.bar_width * np.cos(np.pi * 0.5 - bar_state[id][4])
            p2y = bar_state[id][1] + 0.5 * self.bar_width * np.sin(np.pi * 0.5 - bar_state[id][4])

            p4x = bar_state[id][2] + 0.5 * self.bar_width * np.cos(np.pi * 0.5 - bar_state[id][4])
            p4y = bar_state[id][3] - 0.5 * self.bar_width * np.sin(np.pi * 0.5 - bar_state[id][4])

            p3x = bar_state[id][2] - 0.5 * self.bar_width * np.cos(np.pi * 0.5 - bar_state[id][4])
            p3y = bar_state[id][3] + 0.5 * self.bar_width * np.sin(np.pi * 0.5 - bar_state[id][4])
            self.bars[id].vertices = [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]

        # render
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()


class ArmEnv:
    """
    a arm is n-bar linkage
    state: state of each bar, (angle [rad], length [pixels])
    action size: depends on how many bar, each bar has a rotation action
    rotation action:
        -1 -> rotate with -pi
        +1 -> rotate with pi
    """
    def __init__(self,
                 goal=None,
                 n_bars=2,
                 canvas_width=400,
                 canvas_height=400,):
        self.viewer = None
        self.dt = 0.1  # refresh rate

        # canvas setup
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.origin = np.array([canvas_width / 2.0, canvas_height / 2.0])  # x,y the parent rotation center of first bar

        if goal is None:
            self.goal_cbox = np.array([100, 100, 100, 100], dtype=np.float)  # blue box bbox=(x,y,w,h)
        self.goal_bbox = cbox_to_bbox(self.goal_cbox)

        self.state_size = 2  # (rotation_angle, length) for each bar
        self.action_size = n_bars  # equivalent to how many bars, each bar has 1 rotation DOF
        # state can be observed by user
        self.state = np.zeros([self.action_size, self.state_size])
        self.action_bound = [-1, 1]  # todo, seperate bound for different joint

        self.on_goal_counter = 0
        self.on_goal_min_t = 10  # arm need to stop at target area for a while to make a successful run

        # state for internal usage
        self.bar_state = np.zeros([self.action_size, 6])

        # init bars
        for each in self.state:
            each[0] = np.pi / 6  # rotation angle from 0 degree, take counter clock as positive
            each[1] = canvas_width/2.0/n_bars  # length
        self.update_positions()

    def step(self, action):
        """
        :param action:  shape = [n,], rotation_angle_per_second for each bar
        :return:
            state, rank=2, shape=[n_bars, n_states], n_state=(length, rotation_angle[rad])
        """
        done = False
        reward = 0
        info = {}

        action = np.clip(action, *self.action_bound)

        self.state[:, 0] += action * self.dt
        self.state[:, 0] %= np.pi * 2  # normalize
        self.update_positions()

        # done and reward
        done_x = self.goal_bbox[0] < self.bar_state[-1][2] < self.goal_bbox[2]
        done_y = self.goal_bbox[1] < self.bar_state[-1][3] < self.goal_bbox[3]

        if done_x and done_y:
            reward += 1.
            self.on_goal_counter += 1
            if self.on_goal_counter >= self.on_goal_min_t:
                done = True
        else:
            self.on_goal_counter = 0

        return self.state, reward, done, info

    def reset(self, goal=None):
        self.viewer = None
        if goal is None:
            self.goal_cbox = np.array([100, 100, 80, 80], dtype=np.float)  # blue box bbox=(x,y,w,h)
        self.goal_bbox = cbox_to_bbox(self.goal_cbox)
        self.state[0][0] = np.pi / 6  # rotation angle from 0 degree, take counter clock as positive
        self.state[0][1] = 100.0  # length
        self.state[1][0] = np.pi / 6  # rotation angle from 0 degree, take counter clock as positive
        self.state[1][1] = 100.0  # length
        self.update_positions()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.goal_bbox, n_bars=self.action_size)
        self.viewer.update(self.bar_state)

    def update_positions(self):
        """
        need to update bars one by one, due to dependencies. can not update matrix in one step
        :return:
        """
        n_bars = self.state.shape[0]
        for i in range(n_bars):
            # update start point of bar first
            if i == 0:
                self.bar_state[i][:2] = self.origin
            else:
                self.bar_state[i][:2] = self.bar_state[i - 1][2:2 + 2]  # next bar's start is previous bars's end
            # update end point with
            length = self.state[i][1]
            theta = self.state[i][0]
            self.bar_state[i][2] = self.bar_state[i][0] + length * np.cos(theta)  # x = x_start + length * cos(theta)
            self.bar_state[i][3] = self.bar_state[i][1] + length * np.sin(theta)  # y = y_start + length * sin(theta)
            # update polar coordinates
            self.bar_state[i][4] = self.state[i][0]
            self.bar_state[i][5] = self.state[i][1]

    def sample_rand_action(self):
        return np.random.rand(self.action_size)-0.5    # -5, 5


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        state, reward, done, info = env.step(env.sample_rand_action())
        if done is True:
            break
