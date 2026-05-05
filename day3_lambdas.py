from typing import Callable

# basic lambda
#Lambda is a way to write a small throwaway function in one line.

square: Callable[[int], int] = lambda x: x**2
double: Callable[[int], int] = lambda x: x*2

print (square(5))
print (double(5))

#lambda for sorting
models: list[tuple[str, int]] = [
    ("GPT", 96),
    ("BERT", 12),
    ("Llama", 32),
    ("mistral", 24)
]

models.sort(key=lambda model : model[1])
print("sorted by layers")
for name, layers in models:
    print(f"{name}: {layers} layers")


# Lambda with filter
numbers: list[int] = [1,2,3,4,5,6,7,8,9,10]
evens: list[int] = list(filter(lambda x: x % 2== 0, numbers))
print(f'\nevens: {evens}')

#lambda with map
doubled: list[int] = list(map(lambda x: x*2, numbers))
print(f"doubled: {doubled}")
