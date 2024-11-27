import gymnasium as gym


# extension upon gymnasium blackjack that converts observations to integers
class BlackjackEnv(gym.ObservationWrapper):
    def __init__(self, **kwargs):
        self.env = gym.make(
            "Blackjack-v1",
        )
        super().__init__(self.env)
        self.observation_space = gym

    def observation(self, obs):
        player_hand, dealer_hand, usable_ace = obs
        # we have 10 bits, respectively: 4 bits dealer hand - 5 bits player hand - 1 bit usable acce
        # we put dealer hand at beginning because that does not use all bits
        return player_hand * 2 + dealer_hand * 64 + usable_ace
