# The registry dictionary to store our functions
function_registry = {}


def seaker_register_function(func):
    """
    Decorator that adds the decorated function to
    the function_registry using its name as the key.
    """
    # Use the function's __name__ attribute as the key
    function_registry[func.__name__] = func

    # Return the function as-is so it remains callable
    return func


def get_seaker_registered_function(func: str):
    return function_registry.get(func, None)
