numbers: list[int]= [1,2,3,4,5,6,7,8,9,10]

#Double every number
doubled: list[int] = [num*2 for num in numbers]
print(doubled)
print()

#only even numbers
evens: list [int] = [num for num in numbers if num%2==0]
print (evens)
print()

#square only odd numbers
odd_squares: list[int] = [num**2 for num in numbers if num%2!=0]
print(odd_squares)
print()

#comprehension over a string
sentence: str ="machine learning"
letters: list[str] = [char.upper() for char in sentence if char != " "]
print (letters)
print()

#comprehension over a dictionary
model: dict[str, str] ={
    "name":"GPT",
    "type":"transformer",
    "creator":"OpenAI"
}

keys: list[str] = [key for key in model]
print (keys)