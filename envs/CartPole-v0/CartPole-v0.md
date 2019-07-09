# 1. Solution





## 2. Environment

refs: [openai gym wiki](https://github.com/openai/gym/wiki/CartPole-v0)

### 2.1 Observation

Type: Box(4)

| Num  | Observation          | Min      | Max     |
| ---- | -------------------- | -------- | ------- |
| 0    | Cart Position        | -2.4     | 2.4     |
| 1    | Cart Velocity        | -Inf     | Inf     |
| 2    | Pole Angle           | ~ -41.8° | ~ 41.8° |
| 3    | Pole Velocity At Tip | -Inf     | Inf     |

### 2.2 Actions

Type: Discrete(2)

| Num  | Action                 |
| ---- | ---------------------- |
| 0    | Push cart to the left  |
| 1    | Push cart to the right |

Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

### 2.3 Reward

Reward is 1 for every step taken, including the termination step

### 2.4 Starting State

All observations are assigned a uniform random value between ±0.05

### 2.5 Episode Termination

1. Pole Angle is more than ±12°
2. Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
3. Episode length is greater than 200

### 2.6 Solved Requirements

Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.