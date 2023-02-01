import numpy as np
from utils.env_constants import *


class Q_Learning:
    def __init__(self, discount_factor, learning_rate):
        self.q_values = np.zeros((X_SAMPLE_SIZE, V_SAMPLE_SIZE, NUM_ACTIONS))
        self.policy = np.ones((X_SAMPLE_SIZE, V_SAMPLE_SIZE), dtype=int)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def update_q_value(self, discrete_state, action, new_discrete_state, reward):
        x, v = discrete_state
        td = (
            reward
            + self.discount_factor * np.max(self.q_values[new_discrete_state])
            - self.q_values[x, v, action]
        )
        self.q_values[x, v, action] = (
            self.q_values[x, v, action] + self.learning_rate * td
        )

    def update_policy(self, discrete_state):
        self.policy[discrete_state] = np.argmax(self.q_values[discrete_state])

    def state_values(self):
        return np.amax(self.q_values, axis=2)
