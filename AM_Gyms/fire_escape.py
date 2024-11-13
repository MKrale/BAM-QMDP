# %%
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MEASURE = 4


class FireEscape(gym.Env):
    """ """

    def __init__(self, size=5, fires=3, measure_cost=0.1):
        self.size = size
        self.fires = fires
        self.measure_cost = measure_cost

        self.player = (0, 0)
        self.generate_random_fires()

        # 4 move actions, one measure action to detect fire
        self.action_space = spaces.Discrete(5)
        # observation space is state and whether there is smoke
        # or, state + locations of fire
        # self.observation_space = spaces.OneOf(
        #     spaces.Tuple([spaces.Discrete(self.n * self.n), spaces.Discrete(2)]),
        #     spaces.Tuple([spaces.Discrete(self.smax**self.n), spaces.Discrete(2)]),
        # )

    def int_to_space(self, n: int):
        return (n // self.size, n % self.size)

    def space_to_int(self, space):
        x, y = space
        return x * self.size + y

    def generate_random_fires(self):
        self.fire_locations = np.full((self.size, self.size), False)
        for i in range(self.fires):
            # random fire that is not in initial or final position
            # we disregard the possibility that two fires occur in the same place
            # todo: check if the fires do not block all paths
            n = self.np_random.integers(1, self.size * self.size - 2)
            x, y = self.int_to_space(n)
            self.fire_locations[x][y] = True

    def seed(self, seed=None):
        super().reset(seed=seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, log=True):
        x, y = self.player
        if action == LEFT:
            if x > 0:
                x -= 1
        elif action == DOWN:
            if y > 0:
                y -= 1
        elif action == RIGHT:
            if x < self.size - 1:
                x += 1
        elif action == UP:
            if y < self.size - 1:
                y += 1
        elif action == MEASURE:
            left = down = right = up = False
            if (x > 0) and self.fire_locations[x - 1][y]:
                left = True
            if (y > 0) and self.fire_locations[x][y - 1]:
                down = True
            if (x < self.size - 1) and self.fire_locations[x + 1][y]:
                right = True
            if (y < self.size - 1) and self.fire_locations[x][y + 1]:
                up = True
            return (self.player, (left, down, right, up)), -self.measure_cost, False, {}

        if log:
            self.render()

        self.player = (x, y)
        # detect smoke (i.e. fire in adjacent cell)
        smoke = False
        if (
            ((x > 0) and self.fire_locations[x - 1][y])
            or ((x < self.size - 1) and self.fire_locations[x + 1][y])
            or (y > 0 and self.fire_locations[x][y - 1])
            or (y < self.size - 1 and self.fire_locations[x][y + 1])
        ):
            smoke = True

        if self.fire_locations[x][y]:
            # player is in fire, episode is over, reward = 0
            return (self.player, smoke), 0, True, {}
        if self.player == (self.size - 1, self.size - 1):
            # player is at the end and has won, reward = 1
            return (self.player, smoke), 1, True, {}

        # game not over, regular observation
        return (self.player, smoke), 0, False, {}

    def render(self):
        for y in range(self.size - 1, -1, -1):
            for x in range(self.size):
                if (x, y) == self.player:
                    print("+", end="")
                elif self.fire_locations[x][y]:
                    print("x", end="")
                else:
                    print("_", end="")
            print("")

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.player = (0, 0)
        self.generate_random_fires()

        return self.to_s(self.components)


# %%
