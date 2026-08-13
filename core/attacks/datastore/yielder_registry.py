import inspect


class YielderRegistry:
    @staticmethod
    def register(func):
        """Decorator to mark a method as the primary yield target."""
        func._is_yielder_hook = True
        return func

    @staticmethod
    def get_target_names(instance):
        """Finds the name of the method marked with @register, checking parents if needed."""
        # Iterate through the Method Resolution Order (MRO)
        # Check class definitions directly without accessing the instance
        names = []
        for cls in type(instance).__mro__:
            # Look at all items in the class __dict__
            for name, attr in cls.__dict__.items():
                # Check if it's callable and has the hook marker
                if callable(attr) and hasattr(attr, "_is_yielder_hook"):
                    names.append(name)
        return names

    @staticmethod
    def get_target_name(instance):
        """Finds the name of the method marked with @register, checking parents if needed."""
        # Iterate through the Method Resolution Order (MRO)
        # Check class definitions directly without accessing the instance
        for cls in type(instance).__mro__:
            # Look at all items in the class __dict__
            for name, attr in cls.__dict__.items():
                # Check if it's callable and has the hook marker
                if callable(attr) and hasattr(attr, "_is_yielder_hook"):
                    return name
        return None
