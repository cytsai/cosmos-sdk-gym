import os
import subprocess
import uuid
import atexit
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from utils import StateDict, ReadQueue


class CosmosSDKEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    dtype = np.float32
    action_max = np.nextafter(dtype(1.0), dtype(0.0))

    def __init__(self, env_config={}):
        _env_config = {"sdk_path":"/home/cytsai/research/icf/cosmos-sdk", "sdk_config":"-NumBlocks=2 -BlockSize=100", "verbose":False}
        _env_config.update(env_config)
        self._seed = 42
        self.sdk_path = _env_config["sdk_path"]
        self.sdk_config = _env_config["sdk_config"]
        self.verbose = _env_config["verbose"]
        self.process = None
        self.action_pipe = None
        self.action_data = None
        self.action_space = spaces.Box(0.0, self.action_max, (1,), self.dtype)
        self.observation_space = spaces.Discrete(256)
        self.statedict = StateDict()
        self.readqueue = ReadQueue()
        atexit.register(lambda: self.close())

    def _parse_output(self, collect_reward=True):
        state = self.observation_space.n - 1
        reward = 0.0
        result = ""
        while True:
            line = self.readqueue.readline()
            if self.verbose and line.startswith(("COVERAGE", "STATE", "ACTION", "PASS", "FAIL", "TIMEOUT", "panic")):
                print(line)
            if line.startswith("COVERAGE") and collect_reward:
                coverage = float(line.lstrip("COVERAGE "))
                assert coverage >= self._coverage
                reward = (coverage - self._coverage)
                self._coverage = coverage
            elif line.startswith("STATE"):
                self._range, state = line.lstrip("STATE ").split()
                self._range, state = int(self._range), self.statedict[state] # + self._range]
                break
            elif line.startswith("ACTION"):
                assert eval(line.lstrip("ACTION ")) == eval(self._action[:-1])
            elif line.startswith(("PASS", "FAIL", "TIMEOUT")):
                result = line.split()[0]
                break
            elif line.startswith("panic") and not self._panic:
                self._panic = line
        return state, reward, result

    def _write_result(self, result):
        self.action_data.write("=" * 80 + '\n')
        self.action_data.write(' '.join(self.process.args).replace(".pipe", ".data") + '\n')
        self.action_data.write("COVERAGE " + str(self._coverage) + '\n')
        self.action_data.write(result + ' ' + self._panic + '\n')

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self._seed = seeding.create_seed(seed)
        return [self._seed]

    def close(self):
        self.readqueue.close()
        if self.action_pipe:
            self.action_pipe.close()
            os.remove(self.action_pipe.name)
            self.action_pipe = None
        if self.action_data:
            self.action_data.close()
            self.action_data = None
        if self.process:
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
        self.action_data = os.path.join(fp, fn+".data")
        os.mkfifo(self.action_pipe)
        # 3. launch simulation
        args  = "go test ./simapp/ -run TestFullAppSimulation -Enabled -Commit -v -cover -coverpkg=./... ".split()
        args += self.sdk_config.split() + [f"-Seed={self._seed}", f"-Guide={self.action_pipe}"]
        self.process = subprocess.Popen(args, cwd=self.sdk_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        # 4. open new guide files
        self.action_pipe = open(self.action_pipe, "w")
        self.action_data = open(self.action_data, "w")
        # 5. get initial state
        self.readqueue.open(self.process.stdout)
        self._coverage = 0.0
        self._panic = ""
        self.state, _, result = self._parse_output(collect_reward=False)
        assert not result
        return self.state

    def _step(self, line):
        self._action = line
        self.action_pipe.write(line)
        self.action_data.write(line)
        self.action_pipe.flush()
        self.state, reward, result = self._parse_output()
        done = len(result) > 0
        if done:
            self._write_result(result)
        return self.state, reward, done, {}

    def step(self, action):
        action = action[0]
        if self._range > 0:
            action = int(self._range * action)
            assert 0 <= action < self._range
        else:
            assert 0.0 <= action < 1.0
        return self._step(str(action) + '\n')

    def render(self, mode='human'):
        print(self.state)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        guide = open(sys.argv[1])
    else:
        guide = None

    env = CosmosSDKEnv({"verbose": True})
    env.reset()
    while True:
        if guide:
            line = guide.readline()
            state, reward, done, _ = env._step(line)
        else:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
        if done:
            break
    env.close()
