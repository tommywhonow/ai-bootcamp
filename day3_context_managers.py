from contextlib import contextmanager
from typing import Generator #generator is a function that uses yield to produce a sequence of values.
import time

#Basic context manager - timer
@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    start: float = time.time()
    print(f"{name} started")
    try:
        yield
    finally:
        end: float=time.time()
        print(f"{name} finished in {end - start: .4f} seconds")

#context manager - suprress errors
@contextmanager
def suppress_errors() -> Generator[None, None, None]:
    try:
        yield
    except Exception as e:
        print(f"Error suppressed: {e}")

#Using them
with timer("training loop"):
    time.sleep(0.3)
    print("training...")

print()

with suppress_errors():
    print("about to divide by zero")
    result: float = 10/0
    print ("this never prints")

print ("program continues after error")