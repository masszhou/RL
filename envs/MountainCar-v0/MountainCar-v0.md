# 1. Solution





# 2. Environment

## 2.1 Observation

Type: Box(2)

| Num  | Observation | Min   | Max  |
| ---- | ----------- | ----- | ---- |
| 0    | position    | -1.2  | 0.6  |
| 1    | velocity    | -0.07 | 0.07 |

## 2.2 Actions

Type: Discrete(3)

| Num  | Action     |
| ---- | ---------- |
| 0    | push left  |
| 1    | no push    |
| 2    | push right |

## 2.3 Reward

-1 for each time step, until the goal position of 0.5 is reached. As with MountainCarContinuous v0, there is no penalty for climbing the left hill, which upon reached acts as a wall.

## 2.4 Starting State

Random position from -0.6 to -0.4 with no velocity.

## 2.5 Episode Termination

The episode ends when you reach 0.5 position, or if 200 iterations are reached.

## 2.6 Solved Requirements

None yet specified