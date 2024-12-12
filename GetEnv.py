import numpy as np
import gymnasium as gym

# Environments
from AM_Gyms.NchainEnv import NChainEnv
from AM_Gyms.Loss_Env import Measure_Loss_Env
from AM_Gyms.frozen_lake_v2 import FrozenLakeEnv_v2
from AM_Gyms.Sepsis.SepsisEnv import SepsisEnv
from AM_Gyms.Blackjack import BlackjackEnv
from AM_Gyms.k_out_of_n import KOutOfN
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from AM_Gyms.ActiveMeasurementWrapper import ActiveMeasurementWrapper


def get_env(env_name, env_gen, env_variant, env_size, remake_env_opt, seed=None):
    "Returns ActiveMeasurement env as specified in global (user-specified) vars"

    np.random.seed(seed)

    # if no measure_cost is provided
    default_measure_cost = 0.05

    remake_env = False

    # Basically, just a big messy pile of if/else statements (Not using match for pre 3.10 python users)

    # Loss-environment, called Measure Regret environment in paper.
    if env_name == "Loss":
        env = Measure_Loss_Env()
        InitialState = 0
        default_measure_cost = 0.1
    # Frozen lake environment (includes all variants)
    elif env_name == "Lake":
        InitialState = 0
        default_measure_cost = 0.05
        if env_size == 0:
            print("Using standard size map (4x4)")
            env_size = 4

        if env_gen == "random":
            map_name = None
            desc = generate_random_map(size=env_size)
        elif env_gen == "standard":
            if env_size != 4 and env_size != 8:
                print("Standard map type can only be used for sizes 4 and 8")
            else:
                map_name = "{}x{}".format(env_size, env_size)
                desc = None
        else:
            print("Using random map")
            map_name = None
            desc = generate_random_map(size=env_size)

        if map_name is None and remake_env_opt:
            remake_env = True

        if env_variant == "det":
            env = gym.make(
                "FrozenLake-v1",
                desc=desc,
                map_name=map_name,
                is_slippery=False,
                render_mode="rgb_array",
            )
        elif env_variant == "slippery":
            env = gym.make(
                "FrozenLake-v1",
                desc=desc,
                map_name=map_name,
                is_slippery=True,
                render_mode="rgb_array",
            )
        elif env_variant == "semi-slippery":
            env = FrozenLakeEnv_v2(
                desc=desc, map_name=map_name, render_mode="rgb_array"
            )
        else:  # default = deterministic
            print("Environment var not recognised! (using deterministic variant)")
            env = gym.make(
                "FrozenLake-v1",
                desc=desc,
                map_name=map_name,
                is_slippery=False,
                render_mode="rgb_array",
            )
    # Taxi environment, as used in AMRL-Q paper. Not used in paper
    elif env_name == "Taxi":
        env = gym.make("Taxi-v3", render_mode="rgb_array")
        InitialState = -1
        default_measure_cost = 0.01 / 20
    elif env_name == "CliffWalking":
        env = gym.make("CliffWalking-v0", render_mode="rgb_array", is_slippery=True)
        InitialState = 36
        default_measure_cost = 0.01 / 20
    # Chain environment, as used in AMRL-Q paper. Not used in paper
    elif env_name == "Chain":
        env = NChainEnv(env_size)
        InitialState = 0
        default_measure_cost = 0.01 / 20
    # Sepsis environment, as used in ACNO-paper. Not used in paper
    elif env_name == "Sepsis":
        env = SepsisEnv()
        InitialState = -1
        default_measure_cost = 0.05
    # Standard OpenAI Gym blackjack environment. Not used in paper
    elif env_name == "Blackjack":
        env = BlackjackEnv(render_mode="rgb_array")
        InitialState = -1
        default_measure_cost = 0.05
    elif env_name == "KOutOfN":
        smax = 4
        n = 4
        if env_size != 0:
            n = env_size
        env = KOutOfN(n=n, smax=smax)
        default_measure_cost = 0.05
        InitialState = 0
    else:
        print("Environment {} not recognised, please try again!".format(env_name))
        return

    env = ActiveMeasurementWrapper(env)

    return env, InitialState, default_measure_cost, remake_env
