'''
File for running & gathering data on Active-Measuring algorithms.
For a brief description of how to use it, see the Readme-file in this repo.

'''

######################################################
        ###             Imports                 ###
######################################################

# File structure stuff
import sys
import os
sys.path.append(os.path.join(sys.path[0],"Baselines"))
sys.path.append(os.path.join(sys.path[0],"Baselines", "ACNO_generalised"))

# External modules
import numpy as np
import gym
import matplotlib.pyplot as plt
import time as t
import datetime
import json
import argparse
from typing import List, Optional
import os

# Agents
import Baselines.AMRL_Agent as amrl
from BAM_QMDP import BAM_QMDP
# from ACNO_Planning_old import ACNO_Planner, ACNO_Planner_SemiRobust, ACNO_Planner_Correct

from Baselines.ACNO_generalised.Observe_then_plan_agent import ACNO_Agent_OTP
from Baselines.ACNO_generalised.Observe_while_plan_agent import ACNO_Agent_OWP
from Baselines.DynaQ import QBasic, QOptimistic, QDyna
from ACNO_Planning import ACNO_Planner, ACNO_Planner_Robust, ACNO_Planner_Control_Robust

# Environments
from AM_Gyms.NchainEnv import NChainEnv
from AM_Gyms.Loss_Env import Measure_Loss_Env
from AM_Gyms.frozen_lake_v2 import FrozenLakeEnv_v2
from AM_Gyms.Sepsis.SepsisEnv import SepsisEnv
from AM_Gyms.Blackjack import BlackjackEnv
from AM_Gyms.MachineMaintenance import Machine_Maintenance_Env
from AM_Gyms.frozen_lake import FrozenLakeEnv, generate_random_map, is_valid
from AM_Gyms.AM_Tables import AM_Environment_Explicit, RAM_Environment_Explicit, OptAM_Environment_Explicit

# Environment wrappers
from AM_Gyms.AM_Env_wrapper import AM_ENV as wrapper
from AM_Gyms.AM_Env_wrapper import AM_Visualiser as visualiser
from Baselines.ACNO_generalised.ACNO_ENV import ACNO_ENV

from AM_Gyms.ModelLearner_Robust import ModelLearner_Robust
from AM_Gyms.generic_gym import GenericAMGym

# JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

######################################################
        ###       Parsing Arguments           ###
######################################################

parser = argparse.ArgumentParser(description="Run tests on Active Measuring Algorithms")

# Defining all parser arguments:
parser.add_argument('-algo'             , default = 'AMRL',             help='Algorithm to be tested.')
parser.add_argument('-env'              , default = 'Lake',             help='Environment on which to perform the testing')
parser.add_argument('-env_var'          , default = 'None',             help='Variant of the environment to use (if applicable)')
parser.add_argument('-env_gen'          , default = None,               help='Size of the environment to use (if applicable)')
parser.add_argument('-env_size'         , default = 0,                  help='Size of the environment to use (if applicable)')
parser.add_argument('-m_cost'           , default = -1.0,               help='Cost of measuring (default: use as specified by environment)')
parser.add_argument('-nmbr_eps'         , default = 500,                help='nmbr of episodes per run')
parser.add_argument('-nmbr_runs'        , default = 1,                  help='nmbr of runs to perform')
parser.add_argument('-f'                , default = None,               help='File name (default: generated automatically)')
parser.add_argument('-rep'              , default = './Data/',          help='Repository to store data (default: ./Data')
parser.add_argument('-save'             , default = True,               help='Option to save or not save data.')
parser.add_argument('-alpha_plan'       , default = 1,                  help='Risk-sensitivity factor as used by planner. Negative values are optimistic.')
parser.add_argument('-alpha_measure'    , default = 0,                  help='Risk-sensitivity factor as used for measuring by Control-Robust ATM. Negative values are optimistic, 0 means alpha_plan is copied.')
parser.add_argument('-alpha_real'       , default = 1,                  help='Risk-sensitivity factor as run on. Negative values are best-cases.')
parser.add_argument('-env_remake'       , default=True,                 help='Option to make a new (random) environment each run or not')

# Unpacking for use in this file:
args             = parser.parse_args()
algo_name        = args.algo
env_name         = args.env
env_variant      = args.env_var
env_size         = int(args.env_size)
env_gen          = str(args.env_gen)
MeasureCost      = float(args.m_cost)
nmbr_eps         = int(args.nmbr_eps)
nmbr_runs        = int(args.nmbr_runs)
file_name        = args.f
rep_name         = args.rep
remake_env_opt   = True
if args.env_remake in  ["False", "false"]:
        remake_env_opt = False


alpha_real       = float(args.alpha_real)
alpha_plan       = float(args.alpha_plan)
alpha_measure    = float(args.alpha_measure)
if alpha_measure == 0:
        alpha_measure = alpha_plan

if args.save == "False" or args.save == "false":
        doSave = False
else:
        doSave = True

# Append size & variant to env_namee
env_name_full = env_name
if env_size != 0:
        env_name_full += "_"+env_gen+str(env_size)

if env_variant != 'None':
        env_name_full += "_"+env_variant

def float_to_str(float):
        if float<0:
                start = "-0"
                float = - float
        else:
                start = "0"
        if float < 1:
                float_str = start + str(float)[2:]
        elif float == 1:
                float_str = "1"
        return float_str
     
# Create env_names for planning & running environmens seperately
env_name_plan = env_name_full + "_a" + float_to_str(alpha_plan)
env_name_real = env_name_full + "_a" + float_to_str(alpha_real)
env_name_measure = env_name_full + "_a" + float_to_str(alpha_measure)

if alpha_plan == alpha_measure:
        env_name_full = env_name_full + "_r" + float_to_str(alpha_real) + "_p" + float_to_str(alpha_plan)
else:
        env_name_full = env_name_full + "_r" + float_to_str(alpha_real) + "_p" + float_to_str(alpha_plan) +  "_m" + float_to_str(alpha_measure)


######################################################
        ###     Intitialise Environment        ###
######################################################

# Lake Envs
s_init                          = 0
MeasureCost_Lake_default        = 0.05
MeasureCost_Taxi_default        = 0.01 / 20
MeasureCost_Chain_default       = 0.05
remake_env                      = False
env_folder_name = os.path.join(os.getcwd(), "AM_Gyms", "Learned_Models")

def get_env(seed = None, get_base = False):
        "Returns AM_Env as specified in global (user-specified) vars"
        global MeasureCost
        global remake_env
        global env_size
        global env_full_name
        
        # Required for making robust env through generic-gym class
        has_terminal_state = True
        max_steps = 10_000
        
        np.random.seed(seed)
        
        # Basically, just a big messy pile of if/else statements...
        
        # Loss-environment, called Measure Regret environment in paper.
        if env_name == "Loss":
                env = Measure_Loss_Env()
                StateSize, ActionSize, s_init = 4, 2, 0
                if MeasureCost == -1:
                        MeasureCost = 0.1
                
                # Frozen lake environment (includes all variants)
        elif env_name == "Lake":
                ActionSize, s_init = 4,0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Lake_default
                if env_size == 0:
                        print("Using standard size map (4x4)")
                        env_size = 4
                        StateSize = 4**2
                else:
                        StateSize = env_size**2
                        
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
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                elif env_variant == "slippery":
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=True)
                elif env_variant == "semi-slippery":
                        env = FrozenLakeEnv_v2(desc=desc, map_name=map_name)
                elif env_variant == None:
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                else: #default = deterministic
                        print("Environment var not recognised! (using deterministic variant)")
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                
        # Taxi environment, as used in AMRL-Q paper. Not used in paper           
        elif env_name == "Taxi":
                env = gym.make('Taxi-v3')
                StateSize, ActionSize, s_init = 500, 6, -1
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Taxi_default

        # Chain environment, as used in AMRL-Q paper. Not used in paper
        elif env_name == "Chain":
                if env_size == '10':
                        StateSize = 10
                elif env_size == '20':
                        StateSize = 20
                elif env_size == '30':
                        StateSize = 30
                elif env_size == '50':
                        StateSize = 50
                else: # default
                        print("env_map not recognised!")
                        StateSize = 20
                        
                env = NChainEnv(StateSize)
                ActionSize, s_init = 2, 0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Chain_default
        
        # Sepsis environment, as used in ACNO-paper. Not used in paper
        elif env_name == 'Sepsis':
                env = SepsisEnv()
                StateSize, ActionSize, s_init = 720, 8, -1
                if MeasureCost == -1:
                        MeasureCost = 0.05

        # Standard OpenAI Gym blackjack environment. Not used in paper
        elif env_name == 'Blackjack':
                env = BlackjackEnv()
                StateSize, ActionSize, s_init = 704, 2, -1
                if MeasureCost ==-1:
                        MeasureCost = 0.05

        elif env_name == "Maintenance":
                if env_size == 0:
                        env_size = 8
                env = Machine_Maintenance_Env(N=env_size)
                StateSize, ActionSize, s_init = env_size+3, 2, 0
                if MeasureCost == -1:
                        MeasureCost = 0.01
                has_terminal_state = False
                max_steps = 50
        
        else:
                print("Environment {} not recognised, please try again!".format(env_name))
                return
                
        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)
        args.m_cost = MeasureCost
                        
        if alpha_real != 1 and not get_base:
                env_explicit = get_explicit_env(ENV, env_folder_name, env_name_real, alpha_real)
                P, _Q, R = env_explicit.get_robust_tables()
                ENV = GenericAMGym(P, R, StateSize, ActionSize, MeasureCost,s_init, has_terminal_state, max_steps)
        
        
        
        return ENV

######################################################
        ###     Defining Agents        ###
######################################################

def get_explicit_env(ENV, env_folder_name, env_name, alpha):
        # is_not_uncertain = (alpha == 0.0 or alpha >= 1.0 or alpha <=-1.0)
        # if is_not_uncertain:
        #         env_explicit = AM_Environment_Explicit()
        if alpha > 0:
                env_explicit = RAM_Environment_Explicit()
        elif alpha <0:
                env_explicit = OptAM_Environment_Explicit()
                alpha = - alpha
        try:
                env_explicit.import_model(fileName = env_name, folder = env_folder_name)
        except FileNotFoundError:
                # if is_not_uncertain:
                #         env_explicit.learn_model_AMEnv(ENV, alpha, df=0.90)
                # else:
                env_explicit.learn_robust_model_Env_alpha(ENV, alpha, df=0.90)
                env_explicit.export_model( env_name, env_folder_name )
        return env_explicit

# Both final names and previous/working names are implemented here
def get_agent(seed=None):
        
        ENV = get_env(seed)
        ENV_base = get_env(seed, get_base=True)
        # AMRL-Q, as specified in original paper
        if algo_name == "AMRL":
                agent = amrl.AMRL_Agent(ENV, turn_greedy=True)
                # AMRL-Q, alter so it is completely greedy in last steps.
        elif algo_name == "AMRL_greedy":
                agent = amrl.AMRL_Agent(ENV, turn_greedy=False)
        # BAM_QMDP, named Dyna-ATMQ in paper. Variant with no offline training
        elif algo_name == "BAM_QMDP":
                agent = BAM_QMDP(ENV, offline_training_steps=0)
        # BAM_QMDP, named Dyna-ATMQ in paper. Variant with 25 offline training steps per real step
        elif algo_name == "BAM_QMDP+":
                agent = BAM_QMDP(ENV, offline_training_steps=25)
                
        elif algo_name == "ATM":
                if (alpha_plan != 1):
                        print("WARNING: Automatically set alpha_plan to 1: ATM algorithm cannot use uncertainty in planning.")
                env_plan = get_explicit_env(ENV_base, env_folder_name, env_name_plan, 1)
                agent = ACNO_Planner(ENV, env_plan)
        elif algo_name == "ATM_Robust":
                env_plan = get_explicit_env(ENV_base, env_folder_name, env_name_plan, alpha_plan)
                agent = ACNO_Planner_Robust(ENV, env_plan)
        elif algo_name == "ATM_Control_Robust":
                env_plan = get_explicit_env(ENV_base, env_folder_name, env_name_plan, alpha_plan)
                env_measure = get_explicit_env(ENV_base, env_folder_name, env_name_measure, alpha_measure)
                agent = ACNO_Planner_Control_Robust(ENV, env_plan, env_measure)
        # Observe-while-planning agent from ACNO-paper. We did not get this to work well, so did not include in in paper
        elif algo_name == "ACNO_OWP":
                ENV_ACNO = ACNO_ENV(ENV)
                agent = ACNO_Agent_OWP(ENV_ACNO)
        # Observe-then-plan agent from ACNO-paper. As used in paper, slight alterations made from original
        elif algo_name == "ACNO_OTP":
                ENV_ACNO = ACNO_ENV(ENV)
                agent = ACNO_Agent_OTP(ENV_ACNO)
        # A number of generic RL-agents. We did not include these in the paper.
        # case "DRQN":
        #         agent = DRQN_Agent(ENV)
        elif algo_name == "QBasic":
                agent = QBasic(ENV)
        elif algo_name == "QOptimistic":
                agent = QOptimistic(ENV)
        elif algo_name == "QDyna":
                agent = QDyna(ENV)
        else:
                        print("Agent not recognised, please try again!")
        return agent

######################################################
        ###     Exporting Results       ###
######################################################

# Automatically creates filename is not specified by user
if file_name == None:
        file_name = 'AMData_{}_{}_mc{}.json'.format(algo_name, env_name_full, str(int(float(args.m_cost)*100)).zfill(3))

# Set measurecost if not set by environment.
if args.m_cost == -1:
        args.m_cost == MeasureCost

def PR_to_data(pr_time):
        "Prints timecode as used in datafiles"
        return (datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=pr_time)).strftime("%d%m%Y%H%M%S")

def export_data(rewards, steps, measures,
                avg_reward, avg_steps, avg_measures, t_start):
        "Exports inputted data, as well as user-set variables, to JSON file"
        with open(rep_name+file_name, 'w') as outfile:
                json.dump({
                        'parameters'            :vars(args),
                        'reward_per_eps'        :rewards,
                        'steps_per_eps'         :steps,
                        'measurements_per_eps'  :measures,
                        'reward_avg'           :avg_reward,
                        'steps_avg'             :avg_steps,
                        'measurements_avg'      :avg_measures,
                        'start_time'            :t_start,
                        'current_time'          :t.perf_counter()
                }, outfile, cls=NumpyEncoder)


######################################################
        ###     Running Simulations       ###
######################################################

rewards, steps, measures = np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps))
t_start = 0 + t.perf_counter()
rewards_avg, steps_avg, measures_avg = np.zeros(nmbr_runs), np.zeros(nmbr_runs), np.zeros(nmbr_runs)
print("""
Start running agent with following settings:
Algorithm: {}
Environment: {}
nmbr runs: {}
nmbr episodes per run: {}.
""".format(algo_name, env_name_full, nmbr_runs, nmbr_eps))

agent = get_agent(0)

for i in range(nmbr_runs):
        t_this_start = t.perf_counter()
        (r_tot, rewards[i], steps[i], measures[i]) = agent.run(nmbr_eps, True) 
        rewards_avg[i], steps_avg[i], measures_avg[i] =np.average(rewards[i]), np.average(steps[i]),np.average(measures[i])
        t_this_end = t.perf_counter()
        if doSave:
                avg_rewards, avg_steps, avg_measures = np.average(rewards_avg), np.average(steps_avg),np.average(measures_avg)
                export_data(rewards[:i+1],steps[:i+1],measures[:i+1],
                            avg_rewards, avg_steps, avg_measures, t_start)
        print("Run {0} done with average reward {2}! (in {1} s, with {3} steps and {4} measurements avg.)\n".format(i+1, t_this_end-t_this_start, rewards_avg[i], steps_avg[i], measures_avg[i]))
        if remake_env and i<nmbr_runs-1:
                agent = get_agent(i+1)
print("Agent Done! ({0} runs in {1} s, with average reward {2}, steps {3}, measures {4})\n\n".format(nmbr_runs, t.perf_counter()-t_start, np.average(rewards_avg), np.average(steps_avg),np.average(measures_avg)))
