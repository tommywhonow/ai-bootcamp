#try — run this code
#except — if this specific error happens, do this instead
#finally — always run this, error or not
#raise — deliberately trigger an error

#Basic try/except
try:
    result: float = 10/0
except ZeroDivisionError:
    print("cannot divide by zero")
print()

#cathing multiple errors
def divide(x: int, y: int) ->float:
    try:
        return x/y
    except ZeroDivisionError:
        print ("cannot divide by zero")
        return 0.0
print()

#FileNotFoundError
def read_file(path:str) ->str:
    try:
        with open(path,"r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"file not found: {path}")
        return ""
print()

#KeyError
def get_value(data: dict[str,str], key : str) ->str:
    try:
        return data[key]
    except KeyError:
        print(f"key not found: {key}")
        return ""
print()

# Finally
def load_model(path: str) -> None:
    try:
        print(f"loading model from {path}")
        raise FileNotFoundError
    except FileNotFoundError:
        print("model file missing")
    finally:
        print("cleanup done- always runs")
print()

print(divide(10,2))
print(divide(10,0))
print(read_file("missing.txt"))
print(get_value({"name": "GPT"}, "version"))
load_model("model.pt")


    



    