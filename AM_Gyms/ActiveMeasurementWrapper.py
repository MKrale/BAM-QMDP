import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space
from typing import Callable


# Slightly desaturate an RGB image by blending it with its grayscale version.
def desaturate_rgb(rgb, alpha=0.5):
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    gray_rgb = np.stack((gray, gray, gray), axis=-1)
    desaturated = (1 - alpha) * rgb + alpha * gray_rgb
    return np.clip(desaturated, 0, 1 if rgb.dtype.kind == "f" else 255)


class ActiveMeasurementWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        observation_function: Callable[
            [Space, Space], Space
        ] = lambda observation, measurement: (observation if measurement else None),
        measurement_cost: Callable[[Space], int] | int = 0.05,
        initial_state=-1,
    ):
        """Custom Active Measurement Wrapper

        Classic AM:
        - Provide no observation_function
        - Let measurement_cost be an integer
        then it returns the whole observation if measured for cost of of measurement cost

        For more customization:
        - Let observation_function: observation -> measurement_action -> new observation
            be a custom observation function dependent on the custom measurement action
        - Let measurement_cost be dependent on the measurement function
        """
        super().__init__(env)
        self.observation_function = observation_function
        if type(measurement_cost) is float:
            self.measurement_cost = lambda measurement_action: (
                measurement_cost if measurement_action else 0
            )
        else:
            self.measurement_cost = measurement_cost
        self.initial_state = initial_state
        self.last_step_measured = False

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.last_step_measured = False
        # do not return observation here
        return None, None

    def step(self, action):
        control_action, measurement_action = action
        self.last_step_measured = measurement_action
        observation, reward, terminated, truncated, info = self.env.step(control_action)
        return (
            self.observation_function(observation, measurement_action),
            reward - self.measurement_cost(measurement_action),
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.env.render_mode == "rgb_array":
            img = self.env.render()
            if not self.last_step_measured:
                return desaturate_rgb(img, 0.65)
            else:
                return img
        elif self.env.render_mode in ["ansi", "text"]:
            return f"Measure action {self.last_step_measured}:\n" + self.env.render()
        self.env.render()

    def get_vars(self):
        return (
            self.env.observation_space,
            self.env.action_space,
            self.measurement_cost,
            self.initial_state,
        )
