import os
import numpy as np
import matplotlib.pyplot as plt
from utils.env_constants import *


def discretize_state(state):
    x, v = state[0], state[1]
    return int((x - X_MIN) / X_STEP_SIZE), int((v - V_MIN) / V_STEP_SIZE)


def print_episode(EPISODES, MAX_TIMESTEPS, episode, timesteps):
    episode_str = " " * (len(str(EPISODES - 1)) - len(str(episode))) + str(episode)
    timesteps_str = " " * (len(str(MAX_TIMESTEPS)) - len(str(timesteps))) + str(
        timesteps
    )
    print(
        f"Episode {episode_str}/{EPISODES-1} finished after {timesteps_str} timesteps."
    )


def delete_previous_training_data(OUTPUT_PATH, HEATMAPS_PATH):
    for filename in os.listdir(OUTPUT_PATH):
        if os.path.isfile(os.path.join(OUTPUT_PATH, filename)):
            os.remove(os.path.join(OUTPUT_PATH, filename))
    for filename in os.listdir(HEATMAPS_PATH):
        os.remove(os.path.join(HEATMAPS_PATH, f"{filename}"))


def save_state_values_heatmap(filename, EPISODES, episode, state_values):
    plt.imshow(
        state_values,
        cmap="inferno",
        interpolation="nearest",
        extent=[X_MIN, X_MAX, V_MIN, V_MAX],
        aspect=(X_MAX - X_MIN) / (V_MAX - V_MIN),
    )
    plt.title(f"State-value function (episode {episode}/{EPISODES-1})")
    plt.xticks(np.arange(X_MIN, X_MAX + 0.2, 0.2))
    plt.xlabel("Horizontal position")
    plt.yticks(np.arange(V_MIN, V_MAX, 0.02))
    plt.ylabel("Velocity")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def plot_training_progress(OUTPUT_PATH, episodes_list, timesteps_list):
    plt.plot(episodes_list, timesteps_list)
    plt.title(f"Training progress")
    plt.xlabel("Episodes")
    plt.ylabel("Timesteps")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "training_progress.png"))
    plt.clf()
