
import sys
from tensorflow.python.training import session_run_hook
from numba import cuda

class TestHook(session_run_hook.SessionRunHook):
    def __init__(self):
        self.counter = 0
        self.profile = True

    def after_run(self, run_context, run_values):
        print("====== TestHook: self.counter=%d" % self.counter)
        self.counter += 1

        if self.profile:
            if self.counter == 1:
                print("====== PROFILE START: counter=%d" % self.counter)
                cuda.profile_start()

            if self.counter == 2:
                print("====== PROFILE STOP: counter=%d" % self.counter)
                cuda.profile_stop()
                sys.exit(0)
