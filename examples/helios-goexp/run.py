from collections import defaultdict
import numpy as np
import gym
from env import CosmosSDKEnv
#env = gym.make('cosmos_sdk_gym:CosmosSDK-v0')


class Cell(object):
    def __init__(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0
        self.reward = 0
        self.trajectory = []

    @property
    def score(self):
        """def cntscore(v, w, p):
            e1 = 0.001
            e2 = 0.00001
            return w / ((v + e1) ** p) + e2
        s  = 1.0
        s += cntscore(self.times_chosen,           0.1, 0.5)
        s += cntscore(self.times_chosen_since_new, 0.0, 0.5)
        s += cntscore(self.times_seen,             0.3, 0.5)
        return s"""
        return (self.reward - self.times_chosen) / len(self.trajectory)

    def choose(self):
        self.times_chosen += 1
        self.times_chosen_since_new += 1
        return self.reward, self.trajectory.copy()


def restore(env, trajectory):
    env.reset()
    for action in trajectory:
        _, _, done, _ = env.step(action)
        assert not done
    return env


env = CosmosSDKEnv()
archive = defaultdict(lambda: Cell())

env.reset()
iterations = 0
frames = 0
max_reward = 0
restore_cell = None
reward = 0
trajectory = []

while True:
    found_new_cell = False
    for i in range(np.random.randint(1, 1000) * 100):
        action = env.action_space.sample()
        state, _reward, done, _ = env.step(action)
        trajectory.append(action)
        reward += _reward
        frames += 1
        if reward > max_reward:
            max_reward = reward
            # log trajectory
        if done:
            break

        assert state != 0
        cell = archive[state]
        if cell.times_seen == 0 or len(trajectory) < len(cell.trajectory): # reward == cell.reward
            found_new_cell = True
            cell.times_chosen = 0
            cell.times_chosen_since_new = 0
            cell.reward = reward
            cell.trajectory = trajectory.copy()
        else:
            assert np.isclose(reward, cell.reward)
        cell.times_seen += 1

    if found_new_cell and restore_cell is not None:
        restore_cell.times_chosen_since_new = 0

    scores = np.array([cell.score for cell in archive.values()])
    restore_cell = archive[np.random.choice(list(archive.keys()), p=scores/scores.sum())]
    reward, trajectory = restore_cell.choose()
    env = restore(env, trajectory)

    iterations += 1
    print(f"Iterations: {iterations}, Cells: {len(archive)}, Max Reward: {max_reward}")
