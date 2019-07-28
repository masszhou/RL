[TOC]



* short [summary](./documents/value-based.pdf) about value-based method

# Solved Problems in this Repo
# 1. Gym
refs: [openai gym wiki](https://github.com/openai/gym/wiki/Table-of-environments)

1.1 Classic Control

* CartPole-v1 
  * states space: Box(4, ) 
  * action space: discrete(2)
* Acrobot-v1
  * state space: Box(6, )
  * action space: Discrete(3)
* MountainCar-v0
  * state space: Box(2, )
  * action space: Discrete(3)
* MountainCarContinuous-v0
  * state space: Box(2, )
  * action space: Box(1, )
* Pendulum-v0
  * state space: Box(3, )
  * action space: Box(1, )

1.2 Box2D

* CarRacing-v1

Box2D dependency, install [discussion](https://github.com/openai/gym/issues/100).

```
sudo apt-get install swig
workon your_venv
cd Box2D-py
python setup.py clean
python setup.py install
```

1.3 Atari

* Breakout-v0
* Pong-v0

1.4 MuJoCo

1.5 Others

* gym-TORCS
  * [github](https://github.com/ugo-nama-kun/gym_torcs)
  * [environment instruction](https://arxiv.org/pdf/1304.1672.pdf)
  * [ddpg showcase by Ben Lau](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
  * my implementation using PPO and beta distribution



# 2. Unity
* Unity Banana Navigation
* Unity Reacher to learn gesture
* Unity Tennis

# 3. Self-Made

* 2D arm tracking object
<img src="./imgs/arm2d_test.gif"  width="300" />

* 
