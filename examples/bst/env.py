import os
import subprocess
import atexit
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from cosmos_sdk_gym.envs.utils import StateDict


class BSTEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    dtype = np.float32
    action_max = np.nextafter(dtype(1.0), dtype(0.0))

    def __init__(self, env_config={}):
        self._seed = 42
        self.verbose = False
        self.process = None
        self.action_space = spaces.Discrete(32)
        #self.action_space = spaces.Box(0.0, self.action_max, (1,), self.dtype)
        self.observation_space = spaces.Discrete(32)
        #self.observation_space = spaces.Box(-1, 1, (1,), self.dtype)
        self.statedict = StateDict()
        atexit.register(lambda: self.close())

    def _parse_output(self):
        state = 0
        result = ""
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
                result = line.lstrip("DONE ")
                break
        return state, result
        #return np.array([state / 16.0], dtype=self.dtype), result

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self._seed = seeding.create_seed(seed)
        return [self._seed]

    def close(self):
        if self.process:
            while self.process.poll() is None:
                self.process.stdout.readline()
            self.process.terminate()
            self.process = None

    def reset(self):
        self.close()
        args = "./bst"
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        self.state, result = self._parse_output()
        assert not result
        return self.state

    @staticmethod
    def reward(result):
        array = result.replace('(','').replace(')','').split(',')
        array = [int(n) for n in array if n]
        if len(array) <= 1:
            return 0.0
        if not all(array[i] < array[i+1] for i in range(len(array) - 1)):
            return -1.0
        else:
            #print(result)
            return 1.0

    def step(self, action):
        #action = action[0]
        self._action = action
        self.process.stdin.write(str(action) + '\n')
        self.process.stdin.flush()
        self.state, result = self._parse_output()
        done = len(result) > 0
        if done:
            #print(result)
            reward = self.reward(result)
        else:
            reward = 0.0
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.state)


if __name__ == "__main__":
    env = BSTEnv()
    for i in range(100):
        env.reset()
        while True:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
            else:
                assert state != 0
    env.close()
