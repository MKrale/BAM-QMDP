import gymnasium as gym
from gymnasium import spaces


# extension upon gymnasium blackjack that converts observations to integers
class BlackjackEnv(gym.ObservationWrapper):
    def __init__(self, **kwargs):
        self.env = gym.make(
            "Blackjack-v1",
            **kwargs,
        )
        super().__init__(self.env)
        # 11 possible dealer hand * 32 possible player hand * 2 for usable ace = 704
        self.observation_space = spaces.Discrete(704)
        self.action_space = spaces.Discrete(2)

    def observation(self, obs):
        player_hand, dealer_hand, usable_ace = obs
        # we have 10 bits, respectively: 4 bits dealer hand - 5 bits player hand - 1 bit usable acce
        # we put dealer hand at beginning because that does not use all bits
        return player_hand * 2 + dealer_hand * 64 + usable_ace
