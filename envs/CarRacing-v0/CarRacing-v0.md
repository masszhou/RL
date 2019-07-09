```python
env.spec.max_episode_steps
# Out[124]: 1000
env.spec.nondeterministic
# Out[127]: False
env.spec.reward_threshold
# Out[128]: 900
env.action_space
# Out[130]: Box(3,)
```

* Every action will be repeated for 8 frames.
* To get velocity information, state is defined as adjacent 4 frames in shape (4, 96, 96).

* regression use softplus, classification use tanh

