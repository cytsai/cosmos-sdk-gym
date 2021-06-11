from collections import defaultdict
import json
import threading
import queue


class StateDict(defaultdict):
    def __init__(self, fn="statedict.json", verbose=True):
        self.fn = fn
        self.verbose = verbose
        self.update(self.load())

    def __missing__(self, state):
        self.update(self.load())
        if state in self:
            return self[state]
        elif self.verbose:
            print("NEW STATE", state)

        self[state] = index = len(self) + 1
        with open(self.fn, 'w') as f:
            json.dump(self, f, indent=2)
        return index

    def load(self):
        try:
            with open(self.fn, 'r') as f:
                return json.load(f)
        except:
            return {}


class ReadQueue:
    def __init__(self, bypass=False):
        self.bypass = bypass
        self.source = None
        self.queue = None
        self.thread = None

    def close(self):
        if self.source: 
            self.source.close()
            self.source = None
        if self.queue:
            del self.queue
            self.queue = None
        if self.thread:
            self.thread.join()
            self.thread = None

    def open(self, source):
        def _readline(source, queue):
            try:
                while True:
                    line = source.readline().decode("utf-8").strip()
                    queue.put(line)
            except:
                pass
        self.source = source
        if not self.bypass:
            self.queue = queue.Queue()
            self.thread = threading.Thread(target=_readline, args=(source, self.queue), daemon=True)
            self.thread.start()

    def readline(self, timeout=5):
        if self.bypass:
            line = self.source.readline().decode("utf-8").strip()
        else:
            try:
                line = self.queue.get(timeout=timeout)
            except:
                line = "TIMEOUT"
        return line
