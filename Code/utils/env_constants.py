# Space margins
X_MAX = 0.6
X_MIN = -1.2
V_MAX = 0.07
V_MIN = -0.07

# Size of discrete space
X_STEP_SIZE = 0.005
V_STEP_SIZE = 0.001

X_SAMPLE_SIZE = int((X_MAX - X_MIN) / X_STEP_SIZE) + 1
V_SAMPLE_SIZE = int((V_MAX - V_MIN) / V_STEP_SIZE) + 1

# Number of available actions
NUM_ACTIONS = 3
