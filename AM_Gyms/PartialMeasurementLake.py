from frozen_lake_v2 import FrozenLakeEnv_v2
from ActiveMeasurementWrapper import ActiveMeasurementWrapper

""""
Test file for testing partial measurements. In this instance for Lake to only observe x or y 
"""

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def obs_function(observation, measurement):
    x = observation % 4
    y = observation // 4
    observe_x, observe_y = measurement
    return (x if observe_x else None, y if observe_y else None)


def measurement_cost(measurement):
    observe_x, observe_y = measurement
    cost = 0
    if observe_x:
        cost += 0.05
    if observe_y:
        cost += 0.05
    return cost


env = FrozenLakeEnv_v2(render_mode="human", map_name="4x4")
env = ActiveMeasurementWrapper(
    env, observation_function=obs_function, measurement_cost=measurement_cost
)
