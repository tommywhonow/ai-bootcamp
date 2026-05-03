def add(x:int, y:int) -> int:
    return x+y

def greet(name : str, role: str = "engineer")->str:
    return f"hello {name}, you are a {role}"

def describe_list(items: list[str]) -> None:
    for item in items:
        print (f"- {item}")

print (add(3,5))
print(greet("Tommy"))
print(greet("Tommy","AI engineer"))
describe_list(["python","NumPy","PyTorch"])
