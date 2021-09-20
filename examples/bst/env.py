import os
import subprocess
import uuid
import atexit
import numpy as np
import re
import time
import gym
from gym import spaces
from gym.utils import seeding
from cosmos_sdk_gym.envs.utils import StateDict


class BSTEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, training=True, verbose=False):
        self._seed = 42
        self.training = training
        self.verbose = verbose
        self.process = None
        self.action_space = spaces.Discrete(31)
        self.observation_space = spaces.Discrete(32)
        #self.observation_space = spaces.MultiDiscrete([2]*32)
        self.bitmap = np.zeros(32, dtype=np.int8)
        self.statedict = StateDict()
        self.steps = -1

        os.makedirs("log", exist_ok=True)
        self.log = open("log/" + uuid.uuid4().hex + (".train" if training else ".eval"), 'w')
        def _close():
            self.log.close()
            self.close()
        atexit.register(lambda: _close())

    def _parse_output(self):
        state = 0
        tree = ""
        while True:
            line = self.process.stdout.readline().strip()
            if self.verbose and line.startswith(("STATE", "ACTION", "DONE")):
                print(line)
            if line.startswith("STATE"):
                state = self.statedict[line.lstrip("STATE ")]
                break
            elif line.startswith("ACTION"):
                assert np.isclose(float(line.lstrip("ACTION ")), self._action)
            elif line.startswith("DONE"):
                tree = line.lstrip("DONE ")
                break
        return state, tree
        #self.bitmap[state] = 1
        #return self.bitmap, tree

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self._seed = seeding.create_seed(seed)
        return [self._seed]

    def close(self):
        if self.process:
            self.process.stdin.close()
            while self.process.poll() is None:
                self.process.stdout.readline()
            self.process.terminate()
            self.process = None

    def reset(self):
        self.close()
        args = "./tree 4 0.5"
        self.process = subprocess.Popen(args.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.bitmap.fill(0)
        self.steps = 0
        self.state, tree = self._parse_output()
        assert not tree
        return self.state

    def reward(self, tree):
        array = [int(n) for n in re.findall(r'\d+', tree)]
        assert len(array) > 1
        if all(array[i] < array[i+1] for i in range(len(array) - 1)):
            self.log.write(f"{time.time()} {len(array)}\n")
            return 1.0
        else:
            self.log.write(f"{time.time()} 0\n")
            return 0.0

    def step(self, action):
        self._action = action
        self.process.stdin.write(str(action) + '\n')
        self.process.stdin.flush()
        self.steps += 1
        self.state, tree = self._parse_output()
        done = len(tree) > 0
        if done:
            reward = self.reward(tree)
        else:
            reward = 0.0
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.state)


class FastBSTEnv(BSTEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args = "./tree 4 0.5"
        self.process = subprocess.Popen(args.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    def reset(self):
        if self.steps != 0:
            self.bitmap.fill(0)
            self.steps = 0
            self.state, tree = self._parse_output()
            assert not tree
        return self.state


if __name__ == "__main__":
    env = FastBSTEnv()
    for _ in range(10000):
        env.reset()
        while True:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
    env.close()
