import sys
import ctypes
from cker.program import Variable, new_var, Program


def force_update(frame):
    # This updates function-level 'Fast Locals'
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
    # This ensures global-level changes are synced
    if frame.f_locals is frame.f_globals:
        # In some environments, we must manually push the change
        pass


class AssignHook:
    def __init__(self):
        self.last_vars = {}

    def start(self):
        frame = sys._getframe(1)
        self.last_vars = {k: v for k, v in frame.f_locals.items() if isinstance(v, Variable)}
        sys.settrace(self._trace_callback)
        # Force a refresh immediately
        frame.f_trace = self._trace_callback

    def _trace_callback(self, frame, event, arg):
        # We check on 'line' (pre-execution) but use the data from the PREVIOUS line
        f_locals = frame.f_locals

        for name, old_obj in list(self.last_vars.items()):
            if name in f_locals and f_locals[name] is not old_obj:
                new_val = f_locals[name]

                # Perform the operation
                old_obj.__ilshift__(new_val)

                # RE-ASSIGNMENT
                f_locals[name] = old_obj

                # If we are in global scope, f_locals is f_globals.
                # If we are in a function, we must force the VM to see the change.
                force_update(frame)

        # Update the snapshot for the next line
        self.last_vars = {k: v for k, v in f_locals.items() if isinstance(v, Variable)}
        return self._trace_callback

    def end(self):
        sys.settrace(None)


# --- Test ---
if __name__ == "__main__":
    with Program("sadsdasda") as program:
        ah = AssignHook()
        x = new_var("x")

        ah.start()

        x = 100
        # The moment this line ends, the tracer for the NEXT line
        # (the print) fires and reverts x to the Variable object.
        print(f"Check 1: {x}")

        x = "Hello World"
        print(f"Check 2: {x}")

        ah.end()
