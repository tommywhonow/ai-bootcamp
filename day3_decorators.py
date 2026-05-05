import time
from functools import wraps
from typing import Callable, TypeVar, Any

F= TypeVar("F", bound=Callable[..., Any])

#Decorator 1 - timer
def timer(func: F) -> F:
    @wraps(func)
    def wrapper(*args :Any, **kwargs: Any) -> Any:
        start: float = time.time()
        result = func(*args, **kwargs)
        end: float = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds" )
        return result
    return wrapper # type: ignore[return-value]

# Decorator 2 - logger
def logger(func: F) ->F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"calling {func.__name__} with args={args} and kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} retunred {result}")
        return result
    return wrapper # type: ignore[return-value]

# Applyig decorators
@timer
def slow_function() ->str:
    time.sleep(0.5)
    return"done"

@logger
def add(x: int, y: int) -> int:
    return x+y

@timer
@logger
def multiply(x: int, y: int) -> int:
    return x*y

slow_function()
print()

add(3,5)
print()

multiply(4,6)
print()