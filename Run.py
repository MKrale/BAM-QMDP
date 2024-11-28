"""
File for running & gathering data on Active-Measuring algorithms.
For a brief description of how to use it, see the Readme-file in this repo.
"""

import numpy as np
import time as t
import datetime
import json
import argparse

from GetEnv import get_env
from GetAgent import get_agent


######################################################
###       Parsing Arguments           ###
######################################################

parser = argparse.ArgumentParser(description="Run tests on Active Measuring Algorithms")

# Defining all parser arguments:
parser.add_argument("-algo", default="AMRL", help="Algorithm to be tested.")
parser.add_argument(
    "-env", default="Lake", help="Environment on which to perform the testing"
)
parser.add_argument(
    "-env_var", default="None", help="Variant of the environment to use (if applicable)"
)
parser.add_argument(
    "-env_gen", default=None, help="Size of the environment to use (if applicable)"
)
parser.add_argument(
    "-env_size", default=0, help="Size of the environment to use (if applicable)"
)
parser.add_argument(
    "-m_cost",
    default=-1.0,
    help="Cost of measuring (default: use as specified by environment)",
)
parser.add_argument("-nmbr_eps", default=500, help="Number of episodes per run")
parser.add_argument("-nmbr_runs", default=1, help="Number of runs to perform")
parser.add_argument(
    "-f", default=None, help="File name (default: generated automatically)"
)
parser.add_argument(
    "-rep", default="./Data/", help="Repository to store data (default: ./Data)"
)
parser.add_argument(
    "-env_remake",
    default=True,
    help="Option to make a new (random) environment each run or not",
)
parser.add_argument("-save", default=True, help="Option to save or not save data.")

# Unpacking for use in this file:
args = parser.parse_args()
algo_name = args.algo
env_name = args.env
env_variant = args.env_var
env_size = int(args.env_size)
env_gen = str(args.env_gen)
measure_cost = float(args.m_cost)
nmbr_eps = int(args.nmbr_eps)
nmbr_runs = int(args.nmbr_runs)
file_name = args.f
rep_name = args.rep
remake_env_opt = True

if args.env_remake in ["False", "false", 0]:
    remake_env_opt = False
if args.save in ["False", "false", 0]:
    doSave = False
else:
    doSave = True


######################################################
###     Getting environment and agent       ###
######################################################

env, InitialState, default_measure_cost, remake_env = get_env(
    env_name, env_gen, env_variant, env_size, remake_env_opt, seed=0
)
if measure_cost == -1:
    measure_cost = default_measure_cost
agent = get_agent(env, algo_name, measure_cost, InitialState)


######################################################
###     Exporting Results       ###
######################################################


# JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Create name for Data file
envFullName = env_name
if env_size != 0:
    envFullName += "_" + env_gen + str(env_size)

if env_variant != "None":
    envFullName += "_" + env_variant

# Automatically creates filename is not specified by user
if file_name == None:
    file_name = "AMData_{}_{}_{}.json".format(
        algo_name, envFullName, str(int(float(measure_cost) * 100)).zfill(3)
    )


def PR_to_data(pr_time):
    "Prints timecode as used in datafiles"
    return (
        datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=pr_time)
    ).strftime("%d%m%Y%H%M%S")


def export_data(rewards, steps, measures, t_start):
    "Exports inputted data, as well as user-set variables, to JSON file"
    with open(rep_name + file_name, "w") as outfile:
        json.dump(
            {
                "parameters": vars(args),
                "reward_per_eps": rewards,
                "steps_per_eps": steps,
                "measurements_per_eps": measures,
                "start_time": t_start,
                "current_time": t.perf_counter(),
            },
            outfile,
            cls=NumpyEncoder,
        )


######################################################
###     Running Simulations       ###
######################################################

rewards, steps, measures = (
    np.zeros((nmbr_runs, nmbr_eps)),
    np.zeros((nmbr_runs, nmbr_eps)),
    np.zeros((nmbr_runs, nmbr_eps)),
)
t_start = 0 + t.perf_counter()
rewards_avg, steps_avg, measures_avg = (
    np.zeros(nmbr_runs),
    np.zeros(nmbr_runs),
    np.zeros(nmbr_runs),
)
print(
    """
Start running agent with following settings:
Algorithm: {}
Environment: {}
nmbr runs: {}
nmbr episodes per run: {}.
""".format(
        algo_name, envFullName, nmbr_runs, nmbr_eps
    )
)


for i in range(nmbr_runs):
    t_this_start = t.perf_counter()
    (r_tot, rewards[i], steps[i], measures[i]) = agent.run(nmbr_eps, True)
    rewards_avg[i], steps_avg[i], measures_avg[i] = (
        np.average(rewards[i]),
        np.average(steps[i]),
        np.average(measures[i]),
    )
    t_this_end = t.perf_counter()
    if doSave:
        export_data(rewards[: i + 1], steps[: i + 1], measures[: i + 1], t_start)
    print(
        "Run {0} done with average reward {2}! (in {1} s, with {3} steps and {4} measurements avg.)\n".format(
            i + 1,
            t_this_end - t_this_start,
            rewards_avg[i],
            steps_avg[i],
            measures_avg[i],
        )
    )
    if remake_env and i < nmbr_runs - 1:
        agent = get_agent(i + 1)
print(
    "Agent Done! ({0} runs in {1} s, with average reward {2}, steps {3}, measures {4})\n\n".format(
        nmbr_runs,
        t.perf_counter() - t_start,
        np.average(rewards_avg),
        np.average(steps_avg),
        np.average(measures_avg),
    )
)
