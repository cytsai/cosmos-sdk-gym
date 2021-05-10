import os
import subprocess
import uuid
import numpy as np
from collections import defaultdict
import json
import gym
from gym import spaces
from gym.utils import seeding


class statedict(defaultdict):
    def __init__(self, fn="statedict.json"):
        self.fn = fn
        self.update(self.load())

    def __missing__(self, state):
        print("NEW STATE", state)
        assert self.load() == self
        self[state] = index = len(self)
        with open(self.fn, 'w') as f:
            json.dump(self, f)
        return index

    def load(self):
        try:
            with open(self.fn, 'r') as f:
                return json.load(f)
        except:
            return {}


class CosmosSDKEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, sdk_path="", sdk_config="", verbose=False):
        self._seed = 42
        self.sdk_path = sdk_path
        self.sdk_config = sdk_config
        self.verbose = verbose
        self.process = None
        self.action_pipe = None
        self.action_bkup = None
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        #self.observation_space = spaces.MultiDiscrete
        self.statedict = statedict()

    def _parse_output(self):
        state = ""
        reward = 0.0
        done = False
        while True:
            line = self.process.stdout.readline().decode("utf-8").strip()
            if self.verbose and line.startswith(("COVERAGE", "STATE", "ACTION", "PASS", "FAIL")):
                print(line)
            if line.startswith("COVERAGE"):
                coverage = float(line.lstrip("COVERAGE "))
                assert coverage >= self._coverage
                reward = (coverage - self._coverage)
                self._coverage = coverage
            elif line.startswith("ACTION"):
                assert line.lstrip("ACTION ") == self._action[:-1]
            elif line.startswith("STATE"):
                self._range, state = line.lstrip("STATE ").split()
                self._range, state = int(self._range), self.statedict[state] # + self._range]
                break
            elif line.startswith("PASS"):
                done = True
                break
            elif line.startswith("FAIL"):
                done = True
                reward += 1.0
                break
            #elif line.startswith("coverage"):
            #    coverage = float(re.findall("(\d*(\.\d+)?%)", line)[0][0][:-1]) / 100.0
        return state, reward, done

    def _write_result(self, fail):
        self.action_bkup.write(self.sdk_config + '\n')
        self.action_bkup.write("FAIL\n" if fail else "PASS\n")
        self.action_bkup.write(str(self._coverage) + '\n')

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self._seed = seeding.create_seed(seed)
        return [self._seed]

    def close(self):
        if self.action_pipe:
            self.action_pipe.close()
            os.remove(self.action_pipe.name)
            self.action_pipe = None
        if self.action_bkup:
            self.action_bkup.close()
            self.action_bkup = None
        if self.process:
            #while self.process.poll() is None:
            #    self.process.stdout.readline()
            self.process.terminate()
            self.process = None

    def reset(self):
        # 1. close old guide files
        self.close()
        # 2. prepare new guide files
        fp = os.path.join(os.getcwd(), "guide")
        fn = uuid.uuid4().hex
        os.makedirs(fp, exist_ok=True)
        self.action_pipe = os.path.join(fp, fn+".pipe")
        self.action_bkup = os.path.join(fp, fn+".bkup")
        os.mkfifo(self.action_pipe)
        # 3. launch simulation
        args  = "go test ./simapp/ -run TestFullAppSimulation -Enabled -Commit -v -cover -coverpkg=./... ".split()
        args += self.sdk_config.split() + [f"-Seed={self._seed}", f"-Guide={self.action_pipe}"]
        self.process = subprocess.Popen(args, cwd=self.sdk_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # 4. open new guide files
        self.action_pipe = open(self.action_pipe, "w")
        self.action_bkup = open(self.action_bkup, "w")
        # 5. get initial state
        self._coverage = 0.0
        self.state, _, done = self._parse_output()
        assert not done
        return np.array(self.state)

    def step(self, action):
        if self._range > 0:
            action = int(self._range * action)
            assert 0 <= action < self._range
        else:
            assert 0.0 <= action < 1.0
        self._action = str(action) + '\n'
        self.action_pipe.write(self._action)
        self.action_bkup.write(self._action)
        self.action_pipe.flush()
        self.state, reward, done = self._parse_output()
        if done:
            self._write_result(reward > 1.0)
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.state)
