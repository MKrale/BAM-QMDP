import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Slightly desaturate an RGB image by blending it with its grayscale version.
def desaturate_rgb(rgb, alpha=0.5):
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    gray_rgb = np.stack((gray, gray, gray), axis=-1)
    desaturated = (1 - alpha) * rgb + alpha * gray_rgb
    return np.clip(desaturated, 0, 1 if rgb.dtype.kind == "f" else 255)


class ActiveMeasurementWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, measurement_cost=0.05, initial_state=-1):
        super().__init__(env)
        self.action_space = spaces.Tuple((env.action_space, spaces.Discrete(2)))
        self.measurement_cost = measurement_cost
        self.last_step_measured = False
        self.initial_state = initial_state

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.last_step_measured = False
        # do not return observation here
        return None, None

    def step(self, action):
        control_action, measurement = action
        self.last_step_measured = measurement
        observation, reward, terminated, truncated, info = self.env.step(control_action)
        if measurement:
            return (
                observation,
                reward - self.measurement_cost,
                terminated,
                truncated,
                info,
            )
        else:
            return None, reward, terminated, truncated, info

    def render(self):
        if self.env.render_mode == "rgb_array":
            img = self.env.render()
            if not self.last_step_measured:
                return desaturate_rgb(img, 0.65)
            else:
                return img
        self.env.render()

    def get_vars(self):
        return (
            self.env.observation_space,
            self.env.action_space,
            self.measurement_cost,
            self.initial_state,
        )
