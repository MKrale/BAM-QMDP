import gymnasium as gym

from Baselines.AMRL_Agent import AMRL_Agent as AMRL
from BAM_QMDP import BAM_QMDP

# from Baselines.ACNO_generalised.Observe_then_plan_agent import ACNO_Agent_OTP

# from Baselines.DRQN import DRQN_Agent requires torch to be downloaded, so kept turned off
from Baselines.DynaQ import QBasic, QOptimistic, QDyna

# from Baselines.ACNO_generalised.ACNO_ENV import ACNO_ENV


def get_agent(ENV: gym.Env, algo_name, MeasureCost, InitialState):
    if algo_name == "AMRL":
        agent = AMRL(
            ENV,
            MeasureCost=MeasureCost,
            InitialState=InitialState,
            turn_greedy=True,
        )
    # AMRL-Q, alter so it is completely greedy in last steps.
    elif algo_name == "AMRL_greedy":
        agent = AMRL(
            ENV,
            MeasureCost=MeasureCost,
            InitialState=InitialState,
            turn_greedy=False,
        )
    # BAM_QMDP, named Dyna-ATMQ in paper. Variant with no offline training
    elif algo_name == "BAM_QMDP":
        agent = BAM_QMDP(
            ENV,
            offline_training_steps=0,
            MeasureCost=MeasureCost,
            InitialState=InitialState,
        )
    # BAM_QMDP, named Dyna-ATMQ in paper. Variant with 25 offline training steps per real step
    elif algo_name == "BAM_QMDP+":
        agent = BAM_QMDP(
            ENV,
            offline_training_steps=25,
            MeasureCost=MeasureCost,
            InitialState=InitialState,
        )
    # Observe-then-plan agent from ACNO-paper. As used in paper, slight alterations made from original
    elif algo_name == "ACNO_OTP":
        ENV_ACNO = ACNO_ENV(ENV)
        agent = ACNO_Agent_OTP(ENV_ACNO)
    # A number of generic RL-agents. We did not include these in the paper.
    # elif algo_name == "DRQN":
    #         agent = DRQN_Agent(ENV)
    elif algo_name == "QBasic":
        agent = QBasic(ENV)
    elif algo_name == "QOptimistic":
        agent = QOptimistic(ENV)
    elif algo_name == "QDyna":
        agent = QDyna(ENV)
    else:
        print("Agent {} not recognised, please try again!".format(algo_name))
    return agent
