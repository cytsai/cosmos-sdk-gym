import os
import subprocess
import uuid
import atexit
import time
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from cosmos_sdk_gym.envs.utils import StateDict, ReadQueue


class CosmosSDKEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    sdk_path = "/home/cytsai/research/icf/cosmos-sdk"
    dtype = np.float32
    #action_max = np.nextafter(dtype(1.0), dtype(0.0))

    def __init__(self, verbose=False):
        self._seed = 42
        self.verbose = verbose
        self.process = None
        self.action_pipe = None
        self.action_data = None
        self.action_space = spaces.Discrete(256)
        #self.action_space = spaces.Box(0.0, self.action_max, (1,), self.dtype)
        self.observation_space = spaces.Tuple((spaces.Discrete(255), spaces.Box(0, np.inf, (1,), self.dtype)))
        self.statedict = StateDict()
        self.readqueue = ReadQueue(bypass=True)
        atexit.register(lambda: self.close())

    def _parse_output(self, collect_reward=True):
        _state = 0
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
                self._range, _state = line.lstrip("STATE ").split()
                self._range, _state = int(self._range), self.statedict[_state]
                break
            elif line.startswith("ACTION"):
                assert eval(line.lstrip("ACTION ")) == eval(self._action)
            elif line.startswith(("PASS", "FAIL", "TIMEOUT")):
                result = line.split()[0]
                break
            elif line.startswith("panic") and not self._panic:
                self._panic = line
                result = "FAIL"
                break
        return (_state, np.array([np.log10(self._range + 1, dtype=self.dtype)])), reward, result
        #return (0, np.zeros(1, dtype=self.dtype)), reward, result

    def _write_result(self, result):
        self.action_data.write('=' * 80 + '\n')
        self.action_data.write(' '.join(self.process.args).replace(".pipe", ".data") + '\n')
        self.action_data.write(f"COVERAGE {self._coverage} STEPS {self._steps} TIMESTAMP {time.time()}\n")
        self.action_data.write((result + ' ' + self._panic).strip() + '\n')

    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.seed(seed)
        self._seed = seeding.create_seed(seed)
        return [self._seed]

    def close(self):
        if self.action_pipe:
            self.action_pipe.close()
            os.remove(self.action_pipe.name)
            self.action_pipe = None
        if self.action_data:
            self.action_data.close()
            self.action_data = None
        if self.process:
            while self.process.poll() is None:
                self.readqueue.readline()
            self.readqueue.close()
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
        #args = f"go test ./simapp/ -v -coverpkg=./... -run=TestFullAppSimulation -Enabled -Commit -NumBlocks={50} -BlockSize={20} -Seed={self._seed} -Guide={self.action_pipe}"
        args = f"./simapp.test -test.run=TestFullAppSimulation -Enabled -Commit -NumBlocks={50} -BlockSize={20} -Seed={self._seed} -Guide={self.action_pipe}"
        self.process = subprocess.Popen(args.split(), cwd=self.sdk_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=-1)
        # 4. open new guide files
        self.action_pipe = open(self.action_pipe, "w")
        self.action_data = open(self.action_data, "w")
        # 5. get initial state
        self.readqueue.open(self.process.stdout)
        self._coverage = 0.0
        self._steps = 0
        self._range = 0
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
        self._steps += 1
        #if self._steps > 100000:
        #    result = "TIMEOUT"
        done = len(result) > 0
        if done:
            self._write_result(result)
        return self.state, reward, done, {}

    def step(self, action):
        #line = f"{np.random.randint(self._range) if self._range > 0 else np.random.rand()}\n"
        #return self._step(line)
        action /= self.action_space.n
        if self._range > 0:
            action = int(self._range * action)
            assert 0 <= action < self._range
        else:
            assert 0.0 <= action < 1.0
        return self._step(str(action) + '\n')

    def render(self, mode="human"):
        print(self.state)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        guide = open(sys.argv[1])
    else:
        guide = None

    env = CosmosSDKEnv()
    for i in range(400):
        if not guide:
            env.seed(i)
        env.reset()
        while True:
            if guide:
                line = guide.readline()
                state, reward, done, _ = env._step(line)
            else:
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)
                #line = f"{np.random.randint(env._range) if env._range > 0 else np.random.rand()}\n"
                #state, reward, done, _ = env._step(line)
            if done:
                print(env._coverage)
                break
            else:
                assert state[0] != 0
        if guide:
            break
    env.close()
