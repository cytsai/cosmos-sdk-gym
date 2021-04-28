import os
import re
import subprocess
import uuid
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class CosmosSDKEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, sdk_path="", sdk_config="", verbose=False):
        self.sdk_path = sdk_path
        self.sdk_config = sdk_config
        self.verbose = verbose
        self.process = None
        self.action_pipe = None
        self.action_bkup = None
        self.state = None
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self._seed = 42
        #self.observation_space = spaces.MultiDiscrete
        #self.reset()

    def _parse_output(self):
        state = ""
        done = False
        fail = False
        coverage = 0.0
        while True: #process.poll() is None:
            line = self.process.stdout.readline().decode("utf-8").strip()
            if self.verbose and line.startswith(("STATE", "ACTION", "PASS", "FAIL", "coverage")):
               print(line)
            if line.startswith("STATE"):
                state = list(map(int, line.lstrip("STATE").split()))
                break
            elif line.startswith("PASS"):
                done = True
            elif line.startswith("FAIL"):
                done = True
                fail = True
            elif line.startswith("coverage"):
                coverage = float(re.findall("(\d*(\.\d+)?%)", line)[0][0][:-1]) / 100.0
                break
        reward = fail + coverage
        return state, done, reward

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self._seed = seeding.create_seed(seed)
        return [self._seed]

    def close(self):
        if self.action_pipe: # is not None:
            self.action_pipe.close()
            os.remove(self.action_pipe.name)
            self.action_pipe = None
        if self.action_bkup: # is not None:
            self.action_bkup.close()
            # TODO: remove according to results?
            self.action_bkup = None
        if self.process: # is not None:
            #while self.process.poll() is None:
            #    self.process.stdout.readline()
            self.process.terminate()
            self.process = None

    def reset(self):
        self.close()
        # 
        fp = os.path.join(os.getcwd(), "guide")
        fn = uuid.uuid4().hex
        os.makedirs(fp, exist_ok=True)
        self.action_pipe = os.path.join(fp, fn+".pipe")
        self.action_bkup = os.path.join(fp, fn+".bkup")
        os.mkfifo(self.action_pipe)
        #
        args  = "go test ./simapp/ -run TestFullAppSimulation -Enabled -Commit -v -cover -coverpkg=./... ".split()
        args += self.sdk_config.split() + [f"-Seed={self._seed}", f"-Guide={self.action_pipe}"]
        self.process = subprocess.Popen(args, cwd=self.sdk_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # 
        self.action_pipe = open(self.action_pipe, "w")
        self.action_bkup = open(self.action_bkup, "w")
        # 
        self.state, done, _ = self._parse_output()
        assert not done
        return np.array(self.state)

    def step(self, action):
        assert 0.0 <= action < 1.0
        action = str(int(self.state[0] * action)) + '\n'
        self.action_pipe.write(action)
        self.action_bkup.write(action)
        self.action_pipe.flush()
        self.state, done, reward = self._parse_output()
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.state)
