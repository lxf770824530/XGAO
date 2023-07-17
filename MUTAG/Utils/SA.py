import numpy as np
import matplotlib.pyplot as plt
import math

class Temperature_State(object):
    # 搜索树状态，记录在某一个Node节点下的状态数据，包含当前温度，执行轮次，记录整个搜索树的执行记录

    def __init__(self, initial_temperature):

        # 温度
        self.temperature = initial_temperature
        self.cooling_rate = 0.98
        self.minimal_temperature = 0.01


    def get_temperature(self):
        return self.temperature

    def get_minimal_temperature(self):
        return self.minimal_temperature

    def set_temperature(self, temperature):
        self.temperature = temperature

    def temperature_dropping(self):
        # 降温函数
        self.temperature *= self.cooling_rate

    def is_continue(self):
        # The round index starts from 1 to max round number
        return self.temperature > self.minimal_temperature

    def __repr__(self):
        return "State: {}, temperature: {}, minimal_temperature: {}".format(
            hash(self), self.temperature, self.minimal_temperature)




def Metropolos(prediction_parent, prediction_child, state):
    current_temperature = state.get_temperature()
    min_temperature = state.get_minimal_temperature()  # minimum value of terperature

    if current_temperature >= min_temperature:
        if prediction_child - prediction_parent > 0:
            return True
        else:

            p = math.exp(-(prediction_child - prediction_parent) / current_temperature)
            r = np.random.uniform(low=0, high=1)
            if r < p:
                return True
            else:
                return False



