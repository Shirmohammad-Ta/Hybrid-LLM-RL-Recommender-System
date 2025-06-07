
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

class SimpleRecEnv(gym.Env):
    def __init__(self, user_behavior_simulator):
        super(SimpleRecEnv, self).__init__()
        self.simulator = user_behavior_simulator
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self):
        self.state = self.simulator.get_initial_state()
        return self.state

    def step(self, action):
        reward, next_state, done = self.simulator.simulate_action(self.state, action)
        self.state = next_state
        return self.state, reward, done, {}

def train_rl_agent(env):
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

# Example usage:
# from simulator import UserSimulator
# env = SimpleRecEnv(UserSimulator())
# model = train_rl_agent(env)
