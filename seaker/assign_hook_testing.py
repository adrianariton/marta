import sys


class Variable:
    def __init__(self, value):
        self.value = value

    def __ilshift__(self, other):
        # This is the logic for <<=
        print(f"  [OP]  Performing <<= with {other}")
        self.value = other
        return self

    def __repr__(self):
        return f"Variable({self.value})"


class AssignHook:
    def __init__(self):
        self.before_vars = {}

    def start(self):
        frame = sys._getframe(1)
        self.before_vars = dict(frame.f_locals)
        print("--- Hook Started ---")

    def end(self):
        frame = sys._getframe(1)
        after_vars = frame.f_locals

        # We must iterate over a list of keys because we might modify the dict
        for var_name in list(after_vars.keys()):
            if var_name.startswith("__") or var_name == "ah":
                continue

            current_val = after_vars[var_name]

            # Check if this variable changed since start()
            if var_name not in self.before_vars or self.before_vars[var_name] is not current_val:

                # Check if the variable was ALREADY a Variable type before this line
                # OR if it is being assigned to a Variable type now.
                old_val = self.before_vars.get(var_name)

                if isinstance(old_val, Variable):
                    print(
                        f"  [HOOK] Detected assignment to Variable '{var_name}'. Redirecting to <<="
                    )
                    # Manually trigger the <<= behavior
                    # This effectively turns 'a = b' into 'a <<= b'
                    new_val = old_val.__ilshift__(current_val)

                    # Update the local scope so 'a' points back to the original object
                    after_vars[var_name] = new_val


class MyObject(Variable):
    def __init__(self, val):
        super().__init__(val)


if __name__ == "__main__":
    ah = AssignHook()

    # 1. Setup our variable first
    target_var = Variable("Initial State")

    ah.start()

    # 2. We want this 'target_var = 500' to act like 'target_var <<= 500'
    # Normally, this would DESTROY the Variable object and replace it with an int.
    # Our hook will catch it and fix it.
    target_var = 500

    ah.end()

    print("\n--- Final Results ---")
    print(f"target_var: {target_var} (Type: {type(target_var)})")
