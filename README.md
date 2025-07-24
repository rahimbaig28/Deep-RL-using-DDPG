# DDPG Pendulum Controller 

This project implements a **Deep Deterministic Policy Gradient (DDPG)** reinforcement learning agent to solve the classic **Pendulum-v1** environment using the **Stable-Baselines3** library. The Pendulum task involves applying continuous torque to swing and balance a pendulum upright.

##  Algorithm

**DDPG** is an off-policy, actor-critic algorithm designed for environments with **continuous action spaces**. It combines deterministic policy learning with a Q-function estimator and uses techniques such as:

- Experience replay
- Target networks
- Action noise for exploration

##  Dependencies

Install required packages with:

```bash
pip install gymnasium[classic-control] stable-baselines3[extra] shimmy
```

##  How to Run

```python
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Create environment
env = gym.make("Pendulum-v1")
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize and train agent
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=30000)
model.save("ddpg_pendulum")

# Evaluation
episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Episode {ep+1}: Total reward = {total_reward:.2f}")
```

##  Sample Output

```
Episode 1: Total reward = -116.46
Episode 2: Total reward =  -1.63
Episode 3: Total reward = -123.14
Episode 4: Total reward = -283.04
Episode 5: Total reward =  -2.96
```

> Note: Rewards closer to 0 indicate better control. The minimum possible reward is around -200 to -300.

##  Environment

- **Observation Space:** `[cos(theta), sin(theta), theta_dot]`
- **Action Space:** `[-2, 2]` (continuous torque)
- **Goal:** Keep the pendulum upright and still.

##  References

- Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015). *Continuous control with deep reinforcement learning*. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

###  Author

**Rahim Baig**  
Data Analyst, Community Planning & Advocacy Council  
📫 rahimbaig00332211@gmail.com
