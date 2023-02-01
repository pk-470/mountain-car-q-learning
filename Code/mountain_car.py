import os
import gym
from utils import *


# Training hyperparameters
EPISODES = 200_001
DISCOUNT_FACTOR = 0.999
LEARNING_RATE = 0.5

# Demonstration parameters
PRINT_TRAINING_PROGRESS_EVERY_EPISODE = 1000
SAVE_TRAINING_PROGRESS_EVERY_EPISODE = 500
SAVE_STATE_VALUE_HEATMAP_EVERY_EPISODE = EPISODES // 5

# Output paths
OUTPUT_PATH = "./training_output"
HEATMAPS_PATH = os.path.join(OUTPUT_PATH, "heatmaps")


gym.envs.register(
    id="MountainCarQLearning-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=100000,
)


def initialise_env(show_animation=False):
    if show_animation:
        env = gym.make("MountainCarQLearning-v0", render_mode="human")
    else:
        env = gym.make("MountainCarQLearning-v0")

    return env


def train():
    print()
    print("-------------------------TRAINING-------------------------")

    # Delete any previous training data
    delete_previous_training_data(OUTPUT_PATH, HEATMAPS_PATH)

    episodes_list = []
    timesteps_list = []
    env = initialise_env()
    q_learning = Q_Learning(DISCOUNT_FACTOR, LEARNING_RATE)

    # Training loop
    for episode in range(EPISODES):
        state, _ = env.reset()
        discrete_state = discretize_state(state)
        timesteps = 0
        while True:
            # Choose action
            action = q_learning.policy[discrete_state]
            # Observe new state after action
            new_state, reward, terminated, _, _ = env.step(action)
            if terminated:
                break
            new_discrete_state = discretize_state(new_state)
            # Update the Q value and the policy
            q_learning.update_q_value(
                discrete_state, action, new_discrete_state, reward
            )
            q_learning.update_policy(discrete_state)
            # Move on to the new state
            discrete_state = new_discrete_state
            timesteps += 1

        # Log training progress
        if episode % SAVE_TRAINING_PROGRESS_EVERY_EPISODE == 0:
            episodes_list.append(episode)
            timesteps_list.append(timesteps)

        # Print training progress
        if episode % PRINT_TRAINING_PROGRESS_EVERY_EPISODE == 0:
            if episode == 0:
                MAX_TIMESTEPS = timesteps
            print_episode(EPISODES, MAX_TIMESTEPS, episode, timesteps)

        # Save state-value function heatmaps
        if episode % SAVE_STATE_VALUE_HEATMAP_EVERY_EPISODE == 0:
            save_state_values_heatmap(
                os.path.join(HEATMAPS_PATH, f"ep_{episode}.png"),
                EPISODES,
                episode,
                q_learning.state_values(),
            )

    # Save training progress plot
    plot_training_progress(OUTPUT_PATH, episodes_list, timesteps_list)

    # Save final policy
    policy = q_learning.policy
    np.save(os.path.join(OUTPUT_PATH, "policy.npy"), policy)

    print()

    return policy


def test(policy=None, filename=None):
    print()
    print("-------------------------TESTING--------------------------")

    if policy is None and filename is not None:
        policy = np.load(filename)

    env = initialise_env(show_animation=True)
    state, _ = env.reset()
    discrete_state = discretize_state(state)
    terminated = False
    timesteps = 0

    while True:
        if policy is not None:
            action = policy[discrete_state]
        else:
            action = env.action_space.sample()
        new_state, _, terminated, _, _ = env.step(action)
        if terminated:
            break
        discrete_state = discretize_state(new_state)
        timesteps += 1

    print(f"Goal reached after {timesteps} timesteps.")
    print()


def train_and_test():
    policy = train()
    test(policy=policy)


if __name__ == "__main__":
    # test()
    # for _ in range(10):
    #     test(filename="./training_output/policy.npy")
    train_and_test()
