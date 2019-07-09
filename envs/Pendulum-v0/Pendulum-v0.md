* [source](https://github.com/openai/gym/wiki/Pendulum-v0)
* Observation
    * Type: Box(3)

Num | Observation | Min | Max
---------|----------|---------|----
0 | cos(theta) | -1.0 | 1.0
1 | sin(theta) | -1.0 | 1.0
2 | theta dot | -8.0 | 8.0
 
* Actions
    * Type: Box(1)

Num | Action | Min |Max
---------|----------|---------|----
0|Joint effort|-2.0|2.0

* Reward
The precise equation for reward:

-(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)