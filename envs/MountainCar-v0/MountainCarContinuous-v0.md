# 1. Solution



# 2. Environment

## 2.1 Observation

Type: Box(2)

| Num  | Observation  | Min   | Max  |
| ---- | ------------ | ----- | ---- |
| 0    | Car Position | -1.2  | 0.6  |
| 1    | Car Velocity | -0.07 | 0.07 |

Note that velocity has been constrained to facilitate exploration, but this constraint might be relaxed in a more challenging version.

## 2.2 Actions

Type: Box(1)

| Num  | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 0    | Push car to the left (negative value) or to the right (positive value) |

## 2.3 Reward

Reward is 100 for reaching the target of the hill on the right hand side, minus the squared sum of actions from start to goal.

This reward function raises an exploration challenge, because if the agent does not reach the target soon enough, it will figure out that it is better not to move, and won't find the target anymore.

Note that this reward is unusual with respect to most published work, where the goal was to reach the target as fast as possible, hence favouring a bang-bang strategy.

## 2.4 Starting State

Position between -0.6 and -0.4, null velocity.

## 2.5 Episode Termination

Position equal to 0.5. A constraint on velocity might be added in a more challenging version.

Adding a maximum number of steps might be a good idea.

## 2.6 Solved Requirements

Get a reward over 90. This value might be tuned.