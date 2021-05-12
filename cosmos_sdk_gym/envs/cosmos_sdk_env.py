import os
import subprocess
import atexit
import uuid
import numpy as np
from collections import defaultdict
import json
import gym
from gym import spaces
from gym.utils import seeding


class statedict(defaultdict):
    def __init__(self, fn="statedict.json", verbose=False):
        self.fn = fn
        self.verbose = verbose
        self.update(self.load())

    def __missing__(self, state):
        if self.verbose:
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
    dtype = np.float32
    action_max = np.nextafter(dtype(1.0), dtype(0.0))

    def __init__(self, env_config={"sdk_path":"", "sdk_config":"", "verbose":False}):
        self._seed = 42
        self.sdk_path = env_config["sdk_path"]
        self.sdk_config = env_config["sdk_config"]
        self.verbose = env_config["verbose"]
        self.process = None
        self.action_pipe = None
        self.action_bkup = None
        self.action_space = spaces.Box(0.0, self.action_max, (1,), self.dtype)
        self.observation_space = spaces.Discrete(128)
        self.statedict = statedict(verbose=env_config["verbose"])
        atexit.register(lambda: self.close())

    def _parse_output(self, collect_reward=True):
        state = self.observation_space.n - 1
        reward = 0.0
        done = False
        while True:
            line = self.process.stdout.readline().decode("utf-8").strip()
            if self.verbose and line.startswith(("COVERAGE", "STATE", "ACTION", "PASS", "FAIL")):
                print(line)
            if line.startswith("COVERAGE") and collect_reward:
                coverage = float(line.lstrip("COVERAGE "))
                assert coverage >= self._coverage
                reward = (coverage - self._coverage)
                self._coverage = coverage
            elif line.startswith("ACTION"):
                assert eval(line.lstrip("ACTION ")) == eval(self._action[:-1])
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
        self.state, _, done = self._parse_output(collect_reward=False)
        assert not done
        return self.state

    def step(self, action):
        action = action[0]
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
            if reward > 1.0:
                reward -= 1.0
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.state)
