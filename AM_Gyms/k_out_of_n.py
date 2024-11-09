import gym
from gym import spaces
from gym.utils import seeding


class KOutOfN(gym.Env):
    """k out of n problem
    we start with n components all functioning, denoted as 0
    every step there is some chance, depending on the other components, for components deteriorate, their state to increase by one
    Action consists of n bits to define whether to do nothing (0) or repair (1)

    Rewards are as follows:
    if k out of n components are working (not smax), then reward 1
    repairing a component costs 0.25, a broken component costs 0.5
    """

    def __init__(
        self, n=5, k=3, smax=4, repair_cost=0.25, break_cost=0.5, max_steps=100
    ):
        self.n = n
        self.k = k
        self.smax = smax
        self.repair_cost = repair_cost
        self.break_cost = break_cost
        self.max_steps = max_steps

        # start with all components repaired
        self.components = [0] * self.n
        self.current_step = 0
        # action is per component whether to repair or do nothing
        self.action_space = spaces.MultiBinary(self.n)
        # observation space is per component its value
        # but, observations must be integers
        self.observation_space = spaces.Discrete(self.smax**self.n)

    # list of components to state integer
    def to_s(self, components: list[int]):
        s = 0
        for i in range(self.n):
            s += components[i] * self.smax**i
        return s

    # state integer to list of components
    def to_components(self, s: int):
        components = [0] * self.n
        for i in range(self.n - 1, -1, -1):
            components[i] = s // self.smax**i
            s %= self.smax**i
        return components

    # action integer to list of actions per component
    def to_action(self, a: int):
        action = [0] * self.n
        for i in range(self.n - 1, -1, -1):
            action[i] = a // 2**i
            a %= 2**i
        return action

    def step(self, action, log=False):
        action = self.to_action(action)
        done = False
        self.current_step += 1
        if self.current_step == 100:
            done = True

        # process action, calculate next state
        next_components = [0] * self.n
        for i in range(self.n):
            if action[i] == 1:
                next_components[i] = 0
            elif self.components[i] == self.smax - 1:
                # broken component stays broken
                next_components[i] = self.smax - 1
            else:
                broken_neighbors = (
                    self.components[(i - 1) % self.n] == self.smax - 1
                ) + (self.components[(i - 1) % self.n] == self.smax + 1)
                p_degrade = 0
                if broken_neighbors == 0:
                    p_degrade = 0.2
                elif broken_neighbors == 1:
                    p_degrade = 0.5
                elif broken_neighbors == 2:
                    p_degrade = 0.9

                if self.np_random.random() <= p_degrade:
                    next_components[i] = self.components[i] + 1
                else:
                    next_components[i] = self.components[i]
        self.components = next_components

        # calculating reward
        reward = 0
        functioning_components = 0
        for i in range(self.n):
            if action[i] == 1:
                reward -= self.repair_cost

            if self.components[i] == self.smax - 1:
                reward -= self.break_cost
            else:
                functioning_components += 1
        if functioning_components >= self.k:
            # positive reward for at least k functioning components
            reward += 1

        return self.to_s(self.components), reward, done, {}

    def reset(self, seed=None):
        super().reset(seed=seed)

        # start with all components repaired
        self.components = [0] * self.n
        self.current_step = 0

        return self.to_s(self.components)
