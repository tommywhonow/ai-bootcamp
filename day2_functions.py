# default arguments
def greet(name:str, role:str = "engineer") -> str:
    return f"hello{name}, you are a {role}"

#multiple return values
def get_stats(numbers: list[int]) -> tuple[int, int, int,]:
    return min(numbers), max(numbers), sum(numbers)

# *args
def add(*numbers:int) ->int:
    return sum(numbers)

# **kwargs
def print_config(**settings:str) ->None:
    for key, value in settings.items():
        print(f"{key}:{value}")

# calling them
print(greet("Tommy"))
print(greet("Tommy", "AI engineer"))

low, high,total= get_stats([1,2,3,4,5])
print(f"low={low}, high = {high}, total ={total}")

print(add(1,2))
print(add(1,2,3,4,5))

print_config(model="GPT", device="Cuda", precision="float16")
