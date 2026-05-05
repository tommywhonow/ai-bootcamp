from typing import Generator

#basic generator
def count_up(n:int) -> Generator[int, None, None]:
    for i in range(n):
        yield i

# Generator for ML - simulate loading data one by one
def data_loader(dataset: list[int], batch_size:int) -> Generator[list[int], None, None]:
    for i in range(0, len(dataset), batch_size): 
        yield dataset[i: i+ batch_size]

# Infinite generator
def infinite_counter(start: int=0)  ->Generator[int, None, None]:
    n: int = start
    while True:
        yield n 
        n+=1

# Using the generators
print("count_up: ")
for num in count_up(5):
    print(num)

print("\ndata_loader batches: ")
data: list[int] = list(range(10))
for batch in data_loader(data, batch_size=3):
    print(batch)

print("\ninfinite_encounter (first 5): ")
counter = infinite_counter(10)
for _ in range(5): 
    print(next(counter))

